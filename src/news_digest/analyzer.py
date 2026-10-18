"""Фоновый разбор новости моделью под слотом общей очереди (спека news-digest, требование «Фоновый
разбор не мешает вопросам пользователей»; design.md изменения add-news-digest, решения 4–5).

Одна новость — один слот `gate.slot(priority=RESEARCH, kind=KIND_NEWS)`: пользователи получают слот
между новостями, разбор не регистрируется как вопрос чата, не попадает в `/queue`, лимиты и TTL.
Повтор при нарушении занимает ещё один слот. Вызов модели внедряется (`complete`), поэтому цепочка
«нарушение → повтор → откат на «без разбора»» и освобождение слота проверяются тестами на фейках
(tests/test_news_digest_analyzer.py). Модели не отдаются инструменты; на вход идёт только публичный
текст одной новости (prompts.py). Исход разбора проверяет КОД (rules.validate_analysis).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from llm_queue.rules import KIND_NEWS, RESEARCH
from main_client_settings import disable_thinking_extra_body, judge_temperature_kwargs

from . import prompts, rules

logger = logging.getLogger(__name__)

# Блокирующий вызов модели: сообщения → текст ответа (исключение — модель недоступна).
Complete = Callable[[list[dict]], str]

MAX_ANALYSIS_TOKENS = 700


@dataclass(frozen=True)
class AnalysisOutcome:
    """Итог разбора одной новости: проверенный разбор либо None («разбор недоступен»)."""

    analysis: dict[str, str] | None
    attempts: int
    reason: str = ""  # почему разбора нет (для журнала)
    # Модель не ответила (сеть, тайм-аут, сервер): ответа по существу не было, повтор позже
    # может получиться. Нарушение правил или «разбор не настроен» — окончательный исход.
    transient: bool = False


async def analyze_news(
    news: dict,
    gate,
    complete: Complete,
    base_system_prompt: str | None,
) -> AnalysisOutcome:
    """Разбор одной новости: до `rules.MAX_ATTEMPTS` обращений, каждое под своим слотом.

    Недоступность модели (любое исключение вызова, кроме отмены) — сразу «без разбора», без
    повторов: повторять бессмысленно, а слот нужен пользователям. Отмена задачи пробрасывается.
    """
    problems: tuple[str, ...] = ()
    for attempt in range(1, rules.MAX_ATTEMPTS + 1):
        messages = prompts.build_analysis_messages(news, base_system_prompt, problems)
        try:
            async with gate.slot(priority=RESEARCH, kind=KIND_NEWS) as ctx:
                raw = await ctx.run_thread(complete, messages)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 — любой сбой модели → новость без разбора
            logger.warning(
                "Разбор новости %s недоступен: модель не ответила (%s).",
                news.get("id"),
                type(error).__name__,
            )
            return AnalysisOutcome(
                None, attempt, f"модель недоступна ({type(error).__name__})", transient=True
            )

        validation = rules.validate_analysis(raw, news.get("link", ""))
        step = rules.next_step(attempt, validation)
        if step == rules.ACCEPT:
            return AnalysisOutcome(validation.analysis.as_dict(), attempt)
        if step == rules.FALLBACK:
            logger.info(
                "Разбор новости %s отвергнут кодом после %d попыток: %s.",
                news.get("id"),
                attempt,
                "; ".join(validation.problems),
            )
            return AnalysisOutcome(None, attempt, "; ".join(validation.problems))
        problems = validation.problems
    return AnalysisOutcome(None, rules.MAX_ATTEMPTS, "исчерпаны попытки")  # недостижимо


def make_completer(backend, timeout_seconds: float) -> Complete:
    """Блокирующий вызов модели разбора по настройкам оператора (`providers.news_client`).

    Температура 0 (у Kimi параметр не отправляется), JSON-режим провайдера — подсказка, а не
    гарантия (проверка кодом обязательна и без него); облачные параметры (`thinking`) локальному
    серверу не отправляются."""

    def complete(messages: list[dict]) -> str:
        base: dict = {
            "model": backend.model,
            "messages": messages,
            "max_tokens": MAX_ANALYSIS_TOKENS,
            "timeout": timeout_seconds,
            **judge_temperature_kwargs(backend.provider),
        }
        extra_body = disable_thinking_extra_body(backend.provider)
        if extra_body:
            base["extra_body"] = extra_body
        try:
            response = backend.client.chat.completions.create(
                response_format={"type": "json_object"}, **base
            )
        except Exception as error:  # noqa: BLE001
            # Не все серверы принимают response_format: один повтор без него; прочие сбои
            # (сеть, ключ, тайм-аут) поднимутся заново и будут обработаны как «модель недоступна».
            if getattr(error, "status_code", None) != 400:
                raise
            response = backend.client.chat.completions.create(**base)
        return response.choices[0].message.content or ""

    return complete
