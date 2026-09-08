"""Общий каркас для research-режимов (/research_*) — до этого рефакторинга каждый из
constraints.py, reasoning.py и temperature.py заново определял один и тот же набор
хелперов почти без изменений (сбор статистики по токенам, fallback на reasoning_content,
перехват ошибок OpenAI SDK и перевод их в сообщение на русском, обработчик /cancel).

models.py — исключение: он использует отсюда только extract_usage,
content_or_reasoning_fallback, sum_usage и build_cancel_handler, а run_scenario/
api_error_to_message/format_scenario_stats — нет, т.к. его сообщения об ошибках
намеренно провайдер-нейтральны (вызов может уйти как к DeepSeek, так и к Kimi), а
статистика сценария шире (время ответа, стоимость) — см. докстринг этого модуля.
"""

from __future__ import annotations

import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from telegram.ext import ConversationHandler

from config import MAIN_API_KEY_ENV_VAR, MAIN_CLIENT_LABEL

logger = logging.getLogger(__name__)


def extract_usage(response) -> dict[str, int] | None:
    """Достаёт минимальную статистику по токенам из ответа API, если она есть."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def content_or_reasoning_fallback(message, reasoning_note: str = "") -> str | None:
    """Возвращает видимый content, а если он пуст — обрезанный reasoning_content.

    На моделях с рассуждениями (deepseek-reasoner и т.п.) весь лимит max_tokens может
    целиком уйти на скрытые размышления, оставляя видимый content пустым при
    finish_reason == "length". reasoning_note позволяет вызывающему модулю добавить
    уточнение к тексту (например, что рассуждения могут быть на другом языке, чем сама
    задача, — см. reasoning.py, единственный модуль, где это уточнение нужно).
    """
    content = message.content
    if not content:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            return (
                f"[Модель ещё не начала видимый ответ, вот её рассуждения{reasoning_note}]\n\n"
                + reasoning
            )
    return content


def sum_usage(
    first: dict[str, int] | None, second: dict[str, int] | None
) -> dict[str, int] | None:
    """Складывает статистику по токенам двух вызовов API (для сценариев с несколькими
    обращениями к API за один сценарий, например «сравнить все»).
    """
    if first is None and second is None:
        return None
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a, b = (first or {}).get(key), (second or {}).get(key)
        if a is None and b is None:
            continue
        result[key] = (a or 0) + (b or 0)
    return result or None


def format_scenario_stats(finish_reason: str | None, usage: dict[str, int] | None) -> str:
    """Минимальная статистика по итогам сценария: причина остановки и токены."""
    parts = [f"finish_reason={finish_reason or 'н/д'}"]
    if usage:
        parts.append(f"prompt_tokens={usage.get('prompt_tokens', 'н/д')}")
        parts.append(f"completion_tokens={usage.get('completion_tokens', 'н/д')}")
        parts.append(f"total_tokens={usage.get('total_tokens', 'н/д')}")
    return "📈 " + ", ".join(parts)


def api_error_to_message(exc: Exception, label: str, api_key_env_var: str) -> str | None:
    """Переводит исключение OpenAI SDK в сообщение для пользователя на русском.

    Возвращает None, если exc — не один из перехватываемых типов ошибок API; в этом
    случае вызывающий код (run_scenario) сам логирует exc целиком и возвращает общий
    текст «непредвиденная ошибка».
    """
    if isinstance(exc, AuthenticationError):
        logger.error("Ошибка аутентификации %s API — проверьте %s.", label, api_key_env_var)
        return (
            f"❌ Ошибка авторизации на сервере {label}. "
            "Администратору бота нужно проверить API-ключ."
        )
    if isinstance(exc, RateLimitError):
        logger.warning("Превышен лимит запросов к %s API.", label)
        return (
            f"⏳ Сервис {label} временно перегружен (превышен лимит запросов). "
            "Попробуй, пожалуйста, через минуту."
        )
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        logger.warning("Тайм-аут запроса к %s API.", label)
        return f"⏳ {label} не ответил вовремя. Попробуй отправить запрос ещё раз."
    if isinstance(exc, APIConnectionError):
        logger.error("Не удалось подключиться к %s API.", label)
        return f"🌐 Не получилось подключиться к серверу {label}. Проверь соединение и попробуй позже."
    if isinstance(exc, APIStatusError):
        logger.error("%s API вернул ошибку: %s", label, exc)
        return f"⚠️ Сервер {label} вернул ошибку. Попробуй позже."
    return None


def run_scenario(
    handler_fn,
    *args,
    format_answer,
    label: str = MAIN_CLIENT_LABEL,
    api_key_env_var: str = MAIN_API_KEY_ENV_VAR,
) -> tuple[str | None, str | None, str | None]:
    """Выполняет сценарий, возвращает (текст_ответа, текст_ошибки, статистика).

    При ошибке текст_ответа и статистика отсутствуют (None) — обращения к API не
    случилось или оно не завершилось валидным ответом. format_answer(answer,
    finish_reason) -> str форматирует успешный ответ конкретного сценария (например,
    JSON-парсинг в constraints.py или добавление пометки об обрезке ответа).
    """
    try:
        answer, finish_reason, usage = handler_fn(*args)
    except Exception as exc:  # noqa: BLE001 — последний рубеж, чтобы бот не падал целиком
        message = api_error_to_message(exc, label, api_key_env_var)
        if message is None:
            logger.exception(
                "Непредвиденная ошибка при обращении к %s API (исследование).", label
            )
            message = "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже."
        return None, message, None

    formatted = format_answer(answer, finish_reason)
    stats = format_scenario_stats(finish_reason, usage)
    return formatted, None, stats


def build_cancel_handler(user_data_key: str):
    """Фабрика fallback-обработчика /cancel, общего для всех research-режимов.

    user_data_key — ключ context.user_data, который нужно очистить при отмене (свой на
    каждый режим: research_question, reasoning_task, temperature_task, models_task).
    """

    async def cancel(update, context) -> int:
        """Fallback /cancel — принудительный выход из режима исследования."""
        context.user_data.pop(user_data_key, None)
        await update.message.reply_text("Исследование прервано. Возвращаюсь в обычный режим.")
        return ConversationHandler.END

    return cancel
