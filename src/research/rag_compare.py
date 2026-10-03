"""Команды /research_rag_compare, /research_rag_compare_stop и /research_rag_compare_report —
сравнение ответов `/smart_agent` с выключенным и включённым слоем rag на контрольном наборе
вопросов (design.md изменения add-rag-compare-command). Технический/исследовательский режим:
регистрируется и упоминается в /help только при RESEARCH_ENABLED, как остальные /research_*.

Правила и тексты (проверка набора, разбор оценки, метрики, агрегат, остановка, отчёт) — в
research/rag_compare_eval.py (чистый модуль с тестами); здесь — обращения к SmartAgent и
модели-оценщику, фоновый прогон, сохранение отчётов на диск и обработчики Telegram.

Прогон идёт в фоне (asyncio-задача), а не внутри обработчика: обновления PTB обрабатываются
последовательно, и обработчик, ждущий несколько минут, заблокировал бы бота для всех. Каждый
синхронный вызов SmartAgent.ask()/оценщика — в asyncio.to_thread. Ответы строят СВЕЖИЕ
SmartAgent во временных каталогах (пустая память и профиль, источники рыночных данных
выключены): память и настройки реальных чатов не затрагиваются, а режимы отличаются только
состоянием слоя rag. Одновременно идёт один прогон на весь бот.

Уровни прогона (design.md изменения add-rag-rerank-and-rewrite, решение 6; настройки оператора
RAG_COMPARE_LEVEL/RAG_COMPARE_MODES): `answers` — ответы и оценка по фактам (дополнительно к
«без RAG / с RAG» — ответы в выбранных режимах поиска), `search` — только поиск по режимам
(фильтр, переписывание вопроса) и метрики попадания в ожидаемый документ без вызовов модели
ответа и оценщика. Правила и тексты режимов — research/rag_modes_eval.py.

Быстрый отказ (решение 8): у каждого вопроса есть бюджет времени, а обращения к модели идут
клиентом без автоповторов SDK и ждут не дольше остатка бюджета — при «зависшем» провайдере
вопрос не тянется минутами. Остановка (решение 9): команда отменяет фоновую задачу, а
автоостановка срабатывает после нескольких подряд проваленных вопросов; оба пути завершаются
частичным отчётом.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone

from telegram import Message, Update
from telegram.error import TelegramError
from telegram.ext import CommandHandler, ContextTypes, filters

from agents import rag_context
from agents.smart_agent import LAYER_RAG, SmartAgent
from config import (
    EMBEDDINGS_MODEL,
    MAIN_API_KEY_ENV_VAR,
    MAIN_CLIENT_LABEL,
    MAIN_MODEL,
    RAG_CANDIDATES,
    RAG_COMPARE_CITATION_JUDGE_SYSTEM_PROMPT,
    RAG_COMPARE_JUDGE_MAX_TOKENS,
    RAG_COMPARE_JUDGE_SYSTEM_PROMPT,
    RAG_COMPARE_LEVEL,
    RAG_COMPARE_MAX_CONSECUTIVE_FAILURES,
    RAG_COMPARE_MODES,
    RAG_COMPARE_QUESTION_TIMEOUT_SECONDS,
    RAG_COMPARE_QUESTIONS_FILE,
    RAG_COMPARE_REPORT_DIR,
    RAG_INDEX_DIR,
    RAG_MAX_PER_DOC,
    RAG_MIN_CHUNK_CHARS,
    RAG_MIN_SCORE,
    RAG_RELATIVE_MARGIN,
    RAG_SEARCH_TIMEOUT_SECONDS,
    RAG_SMART_AGENT_STRATEGY,
    RAG_TOP_K,
    REQUEST_TIMEOUT_SECONDS,
    REWRITE_HISTORY_QUESTIONS,
    REWRITE_SEARCH_MODE,
    TELEGRAM_MESSAGE_LIMIT,
)
from providers.main_client import main_client
from providers.rewrite_client import rewrite_backend
from rag import index_store

from . import rag_compare_eval as ev
from . import rag_modes_eval as rm
from ._shared import api_error_to_message

logger = logging.getLogger(__name__)

# Набор вопросов НЕ входит в репозиторий (data/rag_eval/questions.json): он привязан к корпусу
# документов оператора. Путь задаёт RAG_COMPARE_QUESTIONS_FILE.
QUESTIONS_PATH = RAG_COMPARE_QUESTIONS_FILE

_EVAL_CHAT_ID = "rag_compare"
_EVAL_PROFILE = "eval"
_REPORT_SUFFIX = ".json"

# Клиент прогона БЕЗ автоповторов SDK (быстрый отказ): повторы SDK умножали ожидание при
# зависшем провайдере (60 с × 3 попытки на каждый вызов). Копия делит http-клиент с main_client.
_fast_client = main_client.with_options(max_retries=0)

# Состояние единственного прогона на весь бот. Читается и меняется только в потоке цикла
# событий (обработчики и фоновая задача), поэтому блокировка не нужна.
_run_in_progress = False
_run_task: asyncio.Task | None = None
_stop_reason: str | None = None  # выставляет команда остановки перед отменой задачи
_finalizing = False  # цикл вопросов закончен, идёт сохранение и отправка итога
_progress_done = 0
_progress_total = 0
_background_tasks: set[asyncio.Task] = set()  # ссылка, чтобы задачу не собрал сборщик мусора


class ReportError(Exception):
    """Сохранённый отчёт нельзя прочитать — команда отвечает сообщением, а не падает."""


class RunCancelled(Exception):
    """Прогон остановлен: рабочий поток вопроса прекращает новые обращения к модели."""


# --------------------------------------------------------------------------- #
# Бюджет времени вопроса
# --------------------------------------------------------------------------- #


class QuestionBudget:
    """Бюджет времени на ОДИН вопрос (быстрый отказ): общий для всех его этапов; каждое
    обращение к модели ждёт не дольше остатка и не больше REQUEST_TIMEOUT_SECONDS."""

    def __init__(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds

    def remaining(self) -> float:
        return self._deadline - time.monotonic()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def call_timeout(self) -> float:
        return max(min(REQUEST_TIMEOUT_SECONDS, self.remaining()), 0.001)


# --------------------------------------------------------------------------- #
# Ответ одного режима и оценка (синхронно — вызываются через asyncio.to_thread)
# --------------------------------------------------------------------------- #


def _error_reason(exc: Exception) -> str:
    """Компактная причина сбоя для отчёта: для ошибок OpenAI SDK — русский текст, иначе
    тип исключения и сообщение."""
    translated = api_error_to_message(exc, MAIN_CLIENT_LABEL, MAIN_API_KEY_ENV_VAR)
    if translated:
        return translated
    logger.exception("Неожиданный сбой в прогоне сравнения RAG.", exc_info=exc)
    return f"{type(exc).__name__}: {exc}".strip()


def _filter_steps() -> tuple[dict, bool]:
    """Шаги отбора для режимов «с отбором»: рабочие или (если все выключены) эталонные —
    см. rm.filter_settings."""
    return rm.filter_settings(
        RAG_CANDIDATES, RAG_TOP_K, RAG_RELATIVE_MARGIN, RAG_MIN_CHUNK_CHARS, RAG_MAX_PER_DOC
    )


def _mode_agent_kwargs(mode_key: str) -> dict:
    """Параметры SmartAgent для режима поиска (design.md изменения add-rag-rerank-and-rewrite,
    решение 6): настройки рабочей конфигурации, кроме отличающихся. Режимы без отбора —
    кандидатов ровно RAG_TOP_K и нейтральные значения шагов (остаётся только порог, как до
    изменения); режимы без переписывания — без модели переписывания."""
    kwargs: dict = {}
    if rm.uses_filter(mode_key):
        steps, _ = _filter_steps()
        kwargs.update(
            rag_candidates=steps["candidates"],
            rag_relative_margin=steps["relative_margin"],
            rag_min_chunk_chars=steps["min_chunk_chars"],
            rag_max_per_doc=steps["max_per_doc"],
        )
    else:
        kwargs.update(
            rag_candidates=RAG_TOP_K,
            rag_relative_margin=1.0,
            rag_min_chunk_chars=0,
            rag_max_per_doc=RAG_TOP_K,
        )
    if not rm.uses_rewrite(mode_key):
        kwargs["rewrite_backend"] = None
    return kwargs


def run_mode(
    question_text: str,
    rag_enabled: bool,
    timeout: float | None = None,
    retrieval_mode: str | None = None,
) -> ev.ModeResult:
    """Ответ одного режима: свежий SmartAgent во временном каталоге (пустая память, профиль
    без полей, источники MCP выключены), слой rag включён или выключен. Настройки поиска —
    из config.py, как в рабочем режиме. Обращение к модели — клиентом без автоповторов и с
    таймаутом `timeout` (по умолчанию REQUEST_TIMEOUT_SECONDS). Сбой провайдера становится
    причиной в результате, а не исключением."""
    # retrieval_mode (rm.MODE_*) меняет только то, чем режим поиска отличается от рабочей
    # конфигурации (отбор, переписывание), см. _mode_agent_kwargs; None — рабочая конфигурация.
    mode_kwargs = _mode_agent_kwargs(retrieval_mode) if retrieval_mode else {}
    with tempfile.TemporaryDirectory(prefix="rag_compare_") as tmp:
        agent = SmartAgent(
            _EVAL_CHAT_ID,
            client=_fast_client,
            timeout=timeout if timeout is not None else REQUEST_TIMEOUT_SECONDS,
            memory_dir=tmp,
            mcp_moex_dir="",
            mcp_bybit_dir="",
            cbr_enabled=False,
            **mode_kwargs,
        )
        agent.create_profile(_EVAL_PROFILE, {})
        agent.set_layer_enabled(LAYER_RAG, rag_enabled)

        limit = timeout if timeout is not None else REQUEST_TIMEOUT_SECONDS
        started = time.monotonic()
        try:
            # Жёсткий предел по часам: таймаут HTTP-клиента — это пауза между чтениями и не
            # ограничивает общее время (см. ev.call_with_deadline).
            answer = ev.call_with_deadline(lambda: agent.ask(question_text), limit)
        except ev.DeadlineExceeded:
            return ev.ModeResult(
                error=f"провайдер не ответил за {limit:.0f} с (превышен предел ожидания)",
                elapsed=time.monotonic() - started,
            )
        except Exception as exc:  # noqa: BLE001 — сбой одного вопроса не должен ронять прогон
            return ev.ModeResult(error=_error_reason(exc), elapsed=time.monotonic() - started)
        elapsed = time.monotonic() - started

    return ev.ModeResult(
        answer=answer.text,
        elapsed=elapsed,
        prompt_tokens=answer.context_tokens,
        completion_tokens=answer.response_tokens,
        llm_calls=answer.llm_calls,
        warnings=list(answer.warnings),
        rag_sources=[
            {
                "title": s.title,
                "chunk_index": s.chunk_index,
                "score": round(s.score, 3),
                "chunk_id": s.chunk_id,
            }
            for s in answer.rag_sources
        ],
        search_failed=rag_context.FAILURE_WARNING in answer.warnings,
        citations=[
            {
                "quote": c.quote,
                "title": c.source.title,
                "chunk_index": c.source.chunk_index,
                "chunk_id": c.source.chunk_id,
            }
            for c in answer.citations
        ],
        citations_written=answer.citations_written,
        citations_unverified=answer.citations_unverified,
        abstained=answer.abstained,
        fragments=ev.unique_fragments(answer.citations),
    )


def _judge_call(
    user_content: str,
    max_tokens: int,
    timeout: float,
    system_prompt: str = RAG_COMPARE_JUDGE_SYSTEM_PROMPT,
) -> dict | None:
    """Один вызов оценщика. «thinking» отключён, как в research/temperature.py: оценщику нужен
    короткий JSON, а на моделях с рассуждениями весь лимит токенов уходил на скрытые
    размышления (finish_reason == "length", content пуст) — оценка терялась. Пустой content
    всё равно возможен (лимит) — его обрабатывает judge() повтором."""
    response = _fast_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
        timeout=timeout,
        response_format={"type": "json_object"},
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content
    if not content:
        return None
    parsed = json.loads(content)
    return parsed if isinstance(parsed, dict) else None


def _judge_data(
    user_content: str, system_prompt: str, budget: QuestionBudget | None
) -> tuple[dict | None, str | None]:
    """Вызов оценщика с одним повтором на удвоенном лимите токенов при обрезанном JSON или
    пустом ответе — только пока не исчерпан бюджет вопроса. Таймаут каждого вызова — остаток
    бюджета (без бюджета — REQUEST_TIMEOUT_SECONDS). Возвращает (данные, None) или (None,
    причина); разбор содержимого — на вызывающем."""

    def call(max_tokens: int) -> dict | None:
        limit = budget.call_timeout() if budget else REQUEST_TIMEOUT_SECONDS
        # Жёсткий предел по часам вокруг вызова (таймаут клиента его не гарантирует).
        return ev.call_with_deadline(
            lambda: _judge_call(user_content, max_tokens, limit, system_prompt), limit
        )

    data: dict | None = None
    try:
        try:
            data = call(RAG_COMPARE_JUDGE_MAX_TOKENS)
        except json.JSONDecodeError:
            data = None
        if data is None:
            if budget is not None and budget.expired():
                return None, ev.QUESTION_TIMEOUT_REASON
            # Обрезанный JSON или пустой ответ (лимит токенов) — один повтор с удвоенным лимитом.
            data = call(RAG_COMPARE_JUDGE_MAX_TOKENS * 2)
    except json.JSONDecodeError:
        return None, "оценщик вернул невалидный JSON"
    except ev.DeadlineExceeded:
        return None, "оценщик не ответил в пределах времени вопроса"
    except Exception as exc:  # noqa: BLE001 — сбой оценки не должен ронять прогон
        return None, _error_reason(exc)
    return data, None


def judge(
    question: ev.Question, answer: str, budget: QuestionBudget | None = None
) -> tuple[ev.Verdict | None, str | None]:
    """Оценка ответа по чек-листу фактов (вслепую по режиму). Возвращает (оценка, None) или
    (None, причина)."""
    user_content = ev.build_judge_user_content(question.question, question.facts, answer)
    data, reason = _judge_data(user_content, RAG_COMPARE_JUDGE_SYSTEM_PROMPT, budget)
    if reason is not None:
        return None, reason
    verdict = ev.parse_judge_response(data, len(question.facts))
    if verdict is None:
        return None, "оценщик вернул непригодный ответ"
    return verdict, None


def judge_citations(
    question: ev.Question,
    answer: str,
    quotes: list[str],
    fragments: list[str],
    budget: QuestionBudget | None = None,
) -> tuple[ev.CitationVerdict | None, str | None]:
    """Оценка опоры ответа на материалы (вслепую по режиму: в запросе только вопрос, ответ,
    ПРОВЕРЕННЫЕ цитаты и полный текст фрагментов, из которых они взяты). Возвращает (вердикт, None)
    или (None, причина)."""
    user_content = ev.build_citation_judge_user_content(
        question.question, answer, quotes, fragments
    )
    data, reason = _judge_data(user_content, RAG_COMPARE_CITATION_JUDGE_SYSTEM_PROMPT, budget)
    if reason is not None:
        return None, reason
    verdict = ev.parse_citation_judge_response(data)
    if verdict is None:
        return None, "оценщик цитат вернул непригодный ответ"
    return verdict, None


def _answer_and_judge(
    question: ev.Question,
    rag_enabled: bool,
    retrieval_mode: str | None,
    answer_stage: str,
    judge_stage: str,
    budget: QuestionBudget,
    on_stage: Callable[[str], None] | None,
    cancelled: Callable[[], bool] | None,
    citation_stage: str | None = None,
) -> ev.ModeResult:
    """Ответ в одном режиме и его оценка в рамках бюджета вопроса: исчерпанный бюджет помечает
    этап причиной «превышен лимит времени вопроса» без новых обращений к модели. С
    `citation_stage` дополнительно оценивается совпадение смысла ответа и проверенных цитат
    (только если они есть)."""
    if cancelled and cancelled():
        raise RunCancelled
    if on_stage:
        on_stage(answer_stage)
    if budget.expired():
        mode = ev.ModeResult(error=ev.QUESTION_TIMEOUT_REASON)
    else:
        mode = run_mode(question.question, rag_enabled, budget.call_timeout(), retrieval_mode)
    if mode.answer is not None:
        if cancelled and cancelled():
            raise RunCancelled
        if on_stage:
            on_stage(judge_stage)
        if budget.expired():
            mode.judge_error = ev.QUESTION_TIMEOUT_REASON
        else:
            mode.verdict, mode.judge_error = judge(question, mode.answer, budget)
        if citation_stage and mode.citations:
            if cancelled and cancelled():
                raise RunCancelled
            if on_stage:
                on_stage(citation_stage)
            if budget.expired():
                mode.citation_judge_error = ev.QUESTION_TIMEOUT_REASON
            else:
                mode.citation_verdict, mode.citation_judge_error = judge_citations(
                    question,
                    mode.answer,
                    [c["quote"] for c in mode.citations],
                    mode.fragments,
                    budget,
                )
    return mode


def variant_stage(mode_key: str, judge_stage: bool) -> str:
    """Ключ этапа ответа (или оценки) в дополнительном режиме поиска — для прогресса."""
    return f"variant_{'judge' if judge_stage else 'answer'}:{mode_key}"


def stage_label(stage: str) -> str | None:
    """Подпись этапа прогресса, которого нет в ev.STAGE_LABELS: режимы поиска и прогон «только
    поиск». None — подпись берётся из ev.STAGE_LABELS."""
    if stage == STAGE_REWRITE:
        return "переписывание вопроса"
    if stage == STAGE_SEARCH:
        return "поиск по режимам"
    for prefix, template in (
        ("variant_answer:", "ответ, режим поиска «{}»"),
        ("variant_judge:", "оценка ответа, режим поиска «{}»"),
    ):
        if stage.startswith(prefix):
            key = stage[len(prefix) :]
            return template.format(rm.MODE_TITLES.get(key, key))
    return None


def evaluate_question(
    question: ev.Question,
    index_titles: list[str],
    on_stage: Callable[[str], None] | None = None,
    budget_seconds: float = RAG_COMPARE_QUESTION_TIMEOUT_SECONDS,
    cancelled: Callable[[], bool] | None = None,
    variant_modes: list[str] | None = None,
) -> ev.QuestionResult:
    """Оба режима по одному вопросу плюс оценка каждого полученного ответа, в рамках общего
    бюджета времени вопроса (budget_seconds): исчерпанный бюджет помечает оставшиеся этапы
    причиной «превышен лимит времени вопроса» без новых обращений. `variant_modes` — режимы
    поиска (rm.MODE_*), по которым ответ и оценка делаются дополнительно (уровень answers).
    `on_stage` вызывается на входе в каждый этап (из рабочего потока) — по нему строится
    прогресс. `cancelled` — признак остановки прогона: поток, брошенный отменённой задачей, на
    следующем этапе завершается (RunCancelled) и не делает новых обращений к модели."""
    budget = QuestionBudget(budget_seconds)
    if question.expect_abstain:
        # Вопрос вне корпуса: нужен только ответ с RAG (ожидается «не знаю»), без оценки по фактам.
        if cancelled and cancelled():
            raise RunCancelled
        if on_stage:
            on_stage(ev.STAGE_ON_ANSWER)
        if budget.expired():
            on = ev.ModeResult(error=ev.QUESTION_TIMEOUT_REASON)
        else:
            on = run_mode(question.question, True, budget.call_timeout())
        return ev.QuestionResult(question=question, off=ev.ModeResult(), on=on)
    plan = (
        (ev.MODE_OFF, False, ev.STAGE_OFF_ANSWER, ev.STAGE_OFF_JUDGE),
        (ev.MODE_ON, True, ev.STAGE_ON_ANSWER, ev.STAGE_ON_JUDGE),
    )
    modes: dict[str, ev.ModeResult] = {}
    for mode_key, rag_enabled, answer_stage, judge_stage in plan:
        modes[mode_key] = _answer_and_judge(
            question,
            rag_enabled,
            None,
            answer_stage,
            judge_stage,
            budget,
            on_stage,
            cancelled,
            citation_stage=ev.STAGE_ON_CITATIONS if rag_enabled else None,
        )
    variants: dict[str, ev.ModeResult] = {}
    for key in variant_modes or []:
        variants[key] = _answer_and_judge(
            question,
            True,
            key,
            variant_stage(key, judge_stage=False),
            variant_stage(key, judge_stage=True),
            budget,
            on_stage,
            cancelled,
        )
    return ev.QuestionResult(
        question=question,
        off=modes[ev.MODE_OFF],
        on=modes[ev.MODE_ON],
        missing_sources=ev.sources_not_indexed(question.sources, index_titles),
        variants=variants,
    )


# --------------------------------------------------------------------------- #
# Прогон «только поиск» по режимам (без вызовов модели ответа и оценщика)
# --------------------------------------------------------------------------- #

STAGE_REWRITE = "rewrite"
STAGE_SEARCH = "search"


def build_mode_agents(tmp: str, modes: list[str]) -> dict[str, SmartAgent]:
    """По агенту на режим поиска: временная память (`tmp`), пустой профиль, источники MCP
    выключены, слой rag включён; настройки — рабочие, кроме отличающихся (_mode_agent_kwargs).
    Агенты переиспользуются на всех вопросах прогона (состояния между вопросами не несут —
    поиск не пишет ни память, ни историю)."""
    agents: dict[str, SmartAgent] = {}
    for mode in modes:
        agent = SmartAgent(
            f"{_EVAL_CHAT_ID}_{mode}",
            client=_fast_client,
            memory_dir=tmp,
            mcp_moex_dir="",
            mcp_bybit_dir="",
            cbr_enabled=False,
            **_mode_agent_kwargs(mode),
        )
        agent.create_profile(_EVAL_PROFILE, {})
        agents[mode] = agent
    return agents


def evaluate_retrieval(
    question: ev.Question,
    index_titles: list[str],
    agents: dict[str, SmartAgent],
    on_stage: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> rm.QuestionRetrieval:
    """Поиск по каждому режиму для одного вопроса. Если среди режимов есть переписывающие,
    вопрос переписывается ОДИН раз и результат переиспользуется (воспроизводимо и дешевле);
    режим «с контекстом диалога» у вопроса с историей переписывает его ещё раз — с предыдущими
    вопросами из набора (не из памяти агента: изолированные агенты своей истории не имеют).
    Сбой переписывания не проваливает вопрос — режимы ищут по исходному тексту, а причина
    попадает в отчёт. Сбой поиска в режиме — причина в результате этого режима, а не
    исключение. У вопроса без истории «склейка» ищет по самому вопросу (как «как до изменения»),
    а «с контекстом диалога» использует обычное переписывание (как «с переписыванием»)."""
    item = rm.QuestionRetrieval(
        question=question, missing_sources=ev.sources_not_indexed(question.sources, index_titles)
    )
    history = list(question.history)
    history_size, _ = rm.history_questions_setting(REWRITE_HISTORY_QUESTIONS)
    context_history = history[-history_size:] if history else []
    context_mode = rm.MODE_REWRITE_CONTEXT in agents
    plain_modes = [m for m in agents if rm.uses_rewrite(m) and m != rm.MODE_REWRITE_CONTEXT]
    need_context = context_mode and bool(context_history)
    need_plain = bool(plain_modes) or (context_mode and not context_history)

    plain_rewrite: tuple[str | None, str | None] | None = None
    context_rewrite: tuple[str | None, str | None] | None = None
    if need_plain:
        if cancelled and cancelled():
            raise RunCancelled
        if on_stage:
            on_stage(STAGE_REWRITE)
        agent = agents[(plain_modes or [rm.MODE_REWRITE_CONTEXT])[0]]
        started = time.monotonic()
        plain_rewrite = agent.rewrite_question(question.question)
        item.rewrite_seconds = time.monotonic() - started
        item.rewrite_query, item.rewrite_failure = plain_rewrite
    if need_context:
        if cancelled and cancelled():
            raise RunCancelled
        if on_stage:
            on_stage(STAGE_REWRITE)
        started = time.monotonic()
        context_rewrite = agents[rm.MODE_REWRITE_CONTEXT].rewrite_question(
            question.question, context_history
        )
        item.context_rewrite_seconds = time.monotonic() - started
        item.context_rewrite_query, item.context_rewrite_failure = context_rewrite
    if on_stage:
        on_stage(STAGE_SEARCH)
    for mode, agent in agents.items():
        if cancelled and cancelled():
            raise RunCancelled
        text = question.question
        rewrite: tuple[str | None, str | None] | None = None
        mode_history: list[str] | None = None
        if mode == rm.MODE_REWRITE_CONTEXT and need_context:
            rewrite, mode_history = context_rewrite, context_history
        elif rm.uses_rewrite(mode):
            rewrite = plain_rewrite
        elif mode == rm.MODE_CONCAT:
            text = rm.concat_text(history, question.question)
        started = time.monotonic()
        try:
            materials, info = agent.search_materials(text, rewrite, mode_history)
        except Exception as exc:  # noqa: BLE001 — сбой одного режима не должен ронять прогон
            item.modes[mode] = rm.RetrievalResult(
                error=_error_reason(exc), seconds=time.monotonic() - started
            )
            continue
        seconds = time.monotonic() - started
        if materials.warning:
            _, reason = agent.get_rag_status()
            item.modes[mode] = rm.RetrievalResult(
                error=reason or ev.SEARCH_FAILED_REASON, seconds=seconds
            )
            continue
        search_text = info.query if info else None
        if mode == rm.MODE_CONCAT and text != question.question:
            search_text = text
        item.modes[mode] = rm.RetrievalResult(
            chunks=[
                {"title": c.title, "chunk_index": c.chunk_index, "score": c.score}
                for c in materials.chunks
            ],
            candidates=info.candidates if info else 0,
            search_text=search_text,
            seconds=seconds,
            history_used=info.history_used if info else 0,
        )
    return item


# --------------------------------------------------------------------------- #
# Хранение отчётов
# --------------------------------------------------------------------------- #


def save_report(report: dict, report_dir: str = RAG_COMPARE_REPORT_DIR) -> str:
    """Пишет отчёт как `<UTC-время>.json`: сначала во временный файл, затем переименование —
    сбой посреди записи не оставляет битого отчёта. Возвращает путь файла."""
    os.makedirs(report_dir, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + _REPORT_SUFFIX
    path = os.path.join(report_dir, name)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def load_latest_report(report_dir: str = RAG_COMPARE_REPORT_DIR) -> dict | None:
    """Последний сохранённый отчёт (файл с наибольшим именем); None — отчётов ещё нет.
    Повреждённый файл — ReportError."""
    try:
        name = ev.latest_report_name(os.listdir(report_dir))
    except FileNotFoundError:
        return None
    if name is None:
        return None
    path = os.path.join(report_dir, name)
    try:
        with open(path, encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(str(exc)) from exc
    if not isinstance(report, dict) or not isinstance(report.get("questions"), list):
        raise ReportError("неожиданная структура отчёта")
    return report


# --------------------------------------------------------------------------- #
# Фоновый прогон
# --------------------------------------------------------------------------- #


def _settings(level: str = rm.LEVEL_ANSWERS) -> dict:
    filter_steps, filter_reference = _filter_steps()
    history_questions, history_reference = rm.history_questions_setting(REWRITE_HISTORY_QUESTIONS)
    return {
        "level": level,
        # Оценщик цитат видит полный текст фрагментов цитат; в отчётах без ключа — только цитаты.
        "citation_judge": ev.CITATION_JUDGE_FRAGMENTS,
        "strategy": RAG_SMART_AGENT_STRATEGY,
        "top_k": RAG_TOP_K,
        "candidates": RAG_CANDIDATES,
        "min_score": RAG_MIN_SCORE,
        "relative_margin": RAG_RELATIVE_MARGIN,
        "min_chunk_chars": RAG_MIN_CHUNK_CHARS,
        "max_per_doc": RAG_MAX_PER_DOC,
        "filter_steps": filter_steps,
        "filter_reference": filter_reference,
        "history_questions": history_questions,
        "history_reference": history_reference,
        "search_timeout": RAG_SEARCH_TIMEOUT_SECONDS,
        "question_timeout": RAG_COMPARE_QUESTION_TIMEOUT_SECONDS,
        "max_consecutive_failures": RAG_COMPARE_MAX_CONSECUTIVE_FAILURES,
        "model": MAIN_MODEL,
        "embeddings_model": EMBEDDINGS_MODEL,
        "rewrite_provider": rewrite_backend.provider if rewrite_backend else "",
        "rewrite_model": rewrite_backend.model if rewrite_backend else "",
        "rewrite_search_mode": REWRITE_SEARCH_MODE if rewrite_backend else "",
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _plural_questions(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        word = "вопрос"
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        word = "вопроса"
    else:
        word = "вопросов"
    return f"{count} {word}"


async def _edit_progress(message: Message, text: str) -> None:
    try:
        await message.edit_text(text)
    except TelegramError:
        # «message is not modified», лимиты Telegram и т.п. — прогресс не должен ронять прогон.
        logger.debug("Не удалось обновить сообщение с прогрессом.", exc_info=True)


class _ProgressReporter:
    """Обновляет одно сообщение с прогрессом. `post()` можно звать из рабочего потока (правка
    планируется в цикле событий без ожидания); устаревшие правки отбрасываются по номеру, а
    после `finish()` новые игнорируются — доживающий поток остановленного прогона не должен
    перезаписать итоговый текст."""

    def __init__(self, loop: asyncio.AbstractEventLoop, message: Message) -> None:
        self._loop = loop
        self._message = message
        self._seq = 0
        self._sent = 0
        self.closed = False
        self._lock = asyncio.Lock()

    def post(self, text: str) -> None:
        if self.closed:
            return
        self._seq += 1
        asyncio.run_coroutine_threadsafe(self._send(self._seq, text), self._loop)

    async def _send(self, seq: int, text: str) -> None:
        async with self._lock:
            if seq < self._sent:
                return
            self._sent = seq
            await _edit_progress(self._message, text)

    async def finish(self, text: str) -> None:
        self.closed = True
        self._seq += 1
        await self._send(self._seq, text)


async def _send_parts(bot, chat_id: int, text: str) -> None:
    for part in ev.split_text(text, TELEGRAM_MESSAGE_LIMIT):
        await bot.send_message(chat_id=chat_id, text=part)


def _initial_progress(level: str, total: int) -> str:
    if level == rm.LEVEL_SEARCH:
        return f"⏳ Сравнение режимов поиска: обработано 0 из {total} вопросов."
    return ev.progress_text(0, total)


async def _run_comparison(
    bot,
    chat_id: int,
    progress_message: Message,
    questions: list[ev.Question],
    index_titles: list[str],
    level: str = rm.LEVEL_ANSWERS,
    modes: list[str] | None = None,
    unavailable: dict[str, str] | None = None,
    skipped_numbers: list[int] | None = None,
) -> None:
    """Фоновая задача прогона. Заканчивается одним из трёх способов: все вопросы обработаны,
    автоостановка (подряд проваленные вопросы) или остановка командой (отмена задачи) — в двух
    последних случаях сохраняется ЧАСТИЧНЫЙ отчёт. Флаг «прогон идёт» снимается в finally,
    поэтому после любой ошибки можно запустить заново. `level` — rm.LEVEL_*: answers (ответы и
    оценка; `modes` — дополнительные режимы поиска) либо search (только поиск по `modes`);
    `unavailable` — режимы, которые не удалось выполнить, с причиной (в отчёт);
    `skipped_numbers` — номера многоходовых вопросов, пропущенных на уровне answers (в отчёт,
    чтобы команда чтения отчёта отвечала по ним понятно)."""
    global _run_in_progress, _run_task, _stop_reason, _finalizing, _progress_done
    modes = list(modes or [])
    unavailable = dict(unavailable or {})
    search_level = level == rm.LEVEL_SEARCH
    loop = asyncio.get_running_loop()
    reporter = _ProgressReporter(loop, progress_message)
    started_at = _now()
    total = len(questions)
    results: list = []
    stop_reason: str | None = None
    cancel_event = threading.Event()
    tmp_dir = tempfile.mkdtemp(prefix="rag_modes_") if search_level else None
    # Бюджет вопроса растёт с числом дополнительных режимов (каждый — ещё ответ и оценка):
    # 120 с рассчитаны на два режима, а предел нужен против «зависшего» провайдера.
    question_budget = RAG_COMPARE_QUESTION_TIMEOUT_SECONDS * (2 + len(modes)) / 2
    try:
        try:
            agents = (
                await asyncio.to_thread(build_mode_agents, tmp_dir, modes) if search_level else {}
            )
            if search_level:
                first_stage = (
                    STAGE_REWRITE if any(rm.uses_rewrite(m) for m in modes) else STAGE_SEARCH
                )
            else:
                first_stage = ev.STAGES[0]
            for index, question in enumerate(questions, start=1):
                reporter.post(
                    ev.progress_text(
                        index - 1, total, first_stage, question.question, stage_label(first_stage)
                    )
                )

                def on_stage(stage: str, done: int = index - 1, text: str = question.question):
                    reporter.post(ev.progress_text(done, total, stage, text, stage_label(stage)))

                if search_level:
                    results.append(
                        await asyncio.to_thread(
                            evaluate_retrieval,
                            question,
                            index_titles,
                            agents,
                            on_stage,
                            cancel_event.is_set,
                        )
                    )
                else:
                    results.append(
                        await asyncio.to_thread(
                            evaluate_question,
                            question,
                            index_titles,
                            on_stage,
                            question_budget,
                            cancel_event.is_set,
                            modes,
                        )
                    )
                _progress_done = index
                limit = RAG_COMPARE_MAX_CONSECUTIVE_FAILURES
                stop_now = (
                    rm.should_stop(results, limit)
                    if search_level
                    else ev.should_stop(results, limit)
                )
                if stop_now:
                    reason = (
                        "поиск не удался во всех режимах (сервер эмбеддингов недоступен "
                        "или повреждён индекс)"
                        if search_level
                        else "провайдер не отвечает или превышен лимит времени вопроса"
                    )
                    stop_reason = (
                        f"автоостановка: {_plural_questions(limit)} подряд завершились сбоем "
                        f"({reason})"
                    )
                    break
        except asyncio.CancelledError:
            # Остановка командой: отмена доставлена один раз — снимаем её и доводим итог.
            asyncio.current_task().uncancel()
            cancel_event.set()  # брошенный рабочий поток не начнёт следующий этап
            stop_reason = _stop_reason or "по команде пользователя"

        _finalizing = True
        status = ev.STATUS_STOPPED if stop_reason else ev.STATUS_COMPLETED
        index_missing = [p for r in results for p in r.missing_sources]
        if search_level:
            report = rm.build_retrieval_report(
                started_at=started_at,
                finished_at=_now(),
                settings=_settings(level),
                results=results,
                planned=total,
                modes=modes,
                unavailable=unavailable,
                index_missing=index_missing,
                status=status,
                stop_reason=stop_reason,
            )
        else:
            report = ev.build_report(
                started_at=started_at,
                finished_at=_now(),
                settings=_settings(level),
                results=results,
                planned=total,
                index_missing=index_missing,
                status=status,
                stop_reason=stop_reason,
                extra={
                    **(
                        {"variant_modes": modes, "unavailable_modes": unavailable}
                        if modes or unavailable
                        else {}
                    ),
                    **({"skipped_multi_turn": list(skipped_numbers)} if skipped_numbers else {}),
                }
                or None,
            )
        await asyncio.to_thread(save_report, report)
        if stop_reason:
            final_note = f"⛔ Прогон остановлен: обработано {len(results)} из {total} вопросов."
        elif search_level:
            final_note = (
                f"⏳ Сравнение режимов поиска: обработано {total} из {total} вопросов. Готово."
            )
        else:
            final_note = ev.progress_text(total, total) + " Готово."
        await reporter.finish(final_note)
        for text in _report_texts(report):
            await _send_parts(bot, chat_id, text)
    except Exception:  # noqa: BLE001 — сообщаем в чат и освобождаем флаг (finally)
        logger.exception("Прогон сравнения RAG завершился неожиданной ошибкой.")
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Прогон сравнения прервался из-за неожиданной ошибки. Результат не "
                "сохранён. Можно запустить /research_rag_compare заново.",
            )
        except TelegramError:
            logger.exception("Не удалось сообщить о сбое прогона в чат.")
    finally:
        reporter.closed = True
        _run_in_progress = False
        _run_task = None
        _stop_reason = None
        _finalizing = False
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _report_texts(report: dict) -> list[str]:
    """Сообщения итога для отчёта любого вида: сводка и таблица по вопросам; у сравнения
    ответов с дополнительными режимами поиска — ещё блок по режимам."""
    if rm.is_retrieval_report(report):
        return [rm.format_retrieval_summary(report), rm.format_retrieval_table(report)]
    texts = [ev.format_summary(report)]
    variants = rm.format_variants_summary(report)
    if variants:
        texts.append(variants)
    texts.append(ev.format_table(report))
    return texts


# --------------------------------------------------------------------------- #
# Обработчики
# --------------------------------------------------------------------------- #


async def rag_compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/research_rag_compare — проверяет набор и индекс, запускает фоновый прогон и сразу
    отвечает; итог придёт отдельными сообщениями."""
    global _run_in_progress, _run_task, _stop_reason, _finalizing
    global _progress_done, _progress_total

    # Все проверки до первого await синхронные: между проверкой флага и его установкой
    # другой обработчик вклиниться не может.
    if not os.path.isfile(QUESTIONS_PATH):
        await update.message.reply_text(
            f"📭 Контрольный набор вопросов не найден: {QUESTIONS_PATH}.\n"
            "Создайте файл — он не входит в репозиторий, потому что привязан к вашему корпусу "
            "документов: JSON-список объектов с полями question, kind (corpus_specific | "
            "concept | post_cutoff), facts (ожидаемые факты) и sources (префиксы заголовков "
            "документов индекса); вопрос вне корпуса — объект с question и "
            "expect_abstain: true (ожидается ответ «не знаю»). Формат и пример — в README. "
            "Другой путь задаётся переменной "
            "RAG_COMPARE_QUESTIONS_FILE."
        )
        return
    try:
        questions = ev.load_questions(QUESTIONS_PATH)
    except ev.QuestionSetError as exc:
        await update.message.reply_text(f"❌ Контрольный набор вопросов некорректен: {exc}.")
        return

    # Уровень и режимы задаёт оператор (RAG_COMPARE_LEVEL/RAG_COMPARE_MODES); пользователь чата
    # их не выбирает. Недопустимое значение — понятное сообщение, а не прогон «как получится».
    try:
        level = rm.parse_level(RAG_COMPARE_LEVEL)
        selected_modes = rm.parse_modes(RAG_COMPARE_MODES)
    except rm.ModeSettingError as exc:
        await update.message.reply_text(f"❌ Настройки сравнения режимов некорректны: {exc}")
        return
    modes, unavailable = rm.resolve_modes(selected_modes, level, rewrite_backend is not None)
    if level == rm.LEVEL_SEARCH and not modes:
        details = "; ".join(f"{rm.MODE_TITLES[k]}: {r}" for k, r in unavailable.items())
        await update.message.reply_text(
            "❌ Для уровня «только поиск» нет доступных режимов"
            + (f" ({details})" if details else "")
            + ". Проверьте RAG_COMPARE_MODES и REWRITE_PROVIDER."
        )
        return

    # Многоходовые вопросы (с history) на уровне ответов не выполняются: ответ на продолжение
    # без настоящей предыстории несравним, они проверяются уровнем search.
    skipped_numbers: list[int] = []
    if level == rm.LEVEL_ANSWERS:
        standalone, multi_turn = ev.split_multi_turn(questions)
        if multi_turn and not standalone:
            await update.message.reply_text(
                "ℹ️ У всех вопросов набора задана история диалога (поле history). Такие вопросы "
                "проверяются сравнением режимов поиска: задайте RAG_COMPARE_LEVEL=search. "
                "Прогон ответов не начат."
            )
            return
        skipped_numbers = [q.number for q in multi_turn]
        questions = standalone

    # Вопросы вне корпуса (expect_abstain) проверяют режим «не знаю» у ответа, а не поиск по
    # ожидаемым документам, поэтому уровень «только поиск» их не обрабатывает.
    if level == rm.LEVEL_SEARCH:
        abstain_numbers = [q.number for q in questions if q.expect_abstain]
        if abstain_numbers:
            questions = [q for q in questions if not q.expect_abstain]
            if not questions:
                await update.message.reply_text(
                    "ℹ️ В наборе только вопросы вне корпуса (expect_abstain): они проверяются "
                    "уровнем ответов (RAG_COMPARE_LEVEL=answers). Прогон не начат."
                )
                return
            await update.message.reply_text(
                "ℹ️ Вопросы вне корпуса ("
                + ", ".join(str(n) for n in abstain_numbers)
                + ") в сравнении режимов поиска не участвуют — их проверяет уровень ответов."
            )

    if _run_in_progress:
        await update.message.reply_text(
            f"⏳ Прогон уже идёт: обработано {_progress_done} из {_progress_total} вопросов.\n"
            "Итог придёт в чат, который его запустил. Остановить — /research_rag_compare_stop."
        )
        return

    if not index_store.index_exists(RAG_SMART_AGENT_STRATEGY, RAG_INDEX_DIR):
        await update.message.reply_text(
            f"📭 Индекс стратегии «{RAG_SMART_AGENT_STRATEGY}» не построен — режим с RAG без "
            "индекса ничего не измерил бы. Сначала выполните `make index` на хосте бота."
        )
        return

    index_titles = index_store.list_titles(RAG_SMART_AGENT_STRATEGY, RAG_INDEX_DIR)
    unindexed = [q.number for q in questions if ev.sources_not_indexed(q.sources, index_titles)]

    _run_in_progress = True
    _stop_reason = None
    _finalizing = False
    _progress_done, _progress_total = 0, len(questions)
    try:
        if unindexed:
            await update.message.reply_text(
                "⚠️ У вопросов "
                + ", ".join(str(n) for n in unindexed)
                + " ожидаемого документа нет в индексе (он мог быть пропущен при индексации): "
                "они будут помечены в отчёте."
            )
        if skipped_numbers:
            await update.message.reply_text(
                f"ℹ️ Пропущено многоходовых вопросов (с историей диалога): {len(skipped_numbers)} — "
                + ", ".join(str(n) for n in skipped_numbers)
                + ". Ответы на продолжения без настоящей предыстории несравнимы; такие вопросы "
                "проверяются на уровне search (RAG_COMPARE_LEVEL=search)."
            )
        for key, reason in unavailable.items():
            await update.message.reply_text(
                f"⚠️ Режим «{rm.MODE_TITLES[key]}» недоступен: {reason}. Он будет помечен в отчёте."
            )
        progress_message = await update.message.reply_text(
            _initial_progress(level, len(questions))
            + "\nБот остаётся доступен; итог придёт сюда, подробности — "
            "/research_rag_compare_report <номер>, остановить — /research_rag_compare_stop."
        )
        task = asyncio.get_running_loop().create_task(
            _run_comparison(
                context.bot,
                update.effective_chat.id,
                progress_message,
                questions,
                index_titles,
                level,
                modes,
                unavailable,
                skipped_numbers,
            )
        )
    except BaseException:
        _run_in_progress = False
        raise
    _run_task = task
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def rag_compare_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/research_rag_compare_stop — останавливает идущий прогон немедленно: отменяет фоновую
    задачу (начатый вызов LLM доживает в потоке до своего таймаута, результат отбрасывается),
    частичный итог придёт в чат, запустивший прогон."""
    global _stop_reason
    if not _run_in_progress or _run_task is None:
        await update.message.reply_text("Прогон сравнения сейчас не идёт.")
        return
    if _finalizing:
        await update.message.reply_text("Прогон уже завершается — итог придёт в чат запуска.")
        return
    if _stop_reason is not None:
        await update.message.reply_text("Остановка уже запрошена, итог придёт в чат запуска.")
        return
    _stop_reason = "по команде пользователя"
    _run_task.cancel()
    await update.message.reply_text(
        "⛔ Останавливаю прогон. Частичный итог придёт в чат, который его запустил."
    )


async def rag_compare_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/research_rag_compare_report [номер] — последний отчёт или детали вопроса."""
    try:
        report = await asyncio.to_thread(load_latest_report)
    except ReportError:
        logger.exception("Не удалось прочитать последний отчёт сравнения RAG.")
        await update.message.reply_text("⚠️ Не удалось прочитать последний отчёт: файл повреждён.")
        return
    if report is None:
        await update.message.reply_text(
            "📭 Отчётов пока нет. Запустите прогон: /research_rag_compare."
        )
        return

    processed = len(report["questions"])
    planned = report.get("planned", processed)
    retrieval = rm.is_retrieval_report(report)
    if not context.args:
        text = "\n\n".join(_report_texts(report))
    else:
        try:
            number = int(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "👉 Формат: /research_rag_compare_report или "
                "/research_rag_compare_report <номер вопроса>"
            )
            return
        text = (
            rm.format_retrieval_detail(report, number)
            if retrieval
            else ev.format_question_detail(report, number, rm.MODE_TITLES)
        )
        if text is None:
            if number in (report.get("skipped_multi_turn") or []):
                await update.message.reply_text(
                    f"Вопрос {number} многоходовый (с историей диалога): на уровне ответов он не "
                    "выполнялся. Его результаты — в отчёте уровня search "
                    "(RAG_COMPARE_LEVEL=search)."
                )
            elif 1 <= number <= planned:
                await update.message.reply_text(
                    f"Вопрос {number} не был обработан: прогон остановлен, "
                    f"обработано {processed} из {planned}."
                )
            else:
                await update.message.reply_text(
                    f"Вопроса с номером {number} нет: в наборе вопросы с 1 по {planned}."
                )
            return

    for part in ev.split_text(text, TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(part)


def build_rag_compare_handlers() -> list[CommandHandler]:
    private = filters.ChatType.PRIVATE
    return [
        CommandHandler("research_rag_compare", rag_compare_command, filters=private),
        CommandHandler("research_rag_compare_stop", rag_compare_stop_command, filters=private),
        CommandHandler("research_rag_compare_report", rag_compare_report_command, filters=private),
    ]
