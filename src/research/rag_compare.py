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
    RAG_COMPARE_JUDGE_MAX_TOKENS,
    RAG_COMPARE_JUDGE_SYSTEM_PROMPT,
    RAG_COMPARE_MAX_CONSECUTIVE_FAILURES,
    RAG_COMPARE_QUESTION_TIMEOUT_SECONDS,
    RAG_COMPARE_QUESTIONS_FILE,
    RAG_COMPARE_REPORT_DIR,
    RAG_INDEX_DIR,
    RAG_MIN_SCORE,
    RAG_SEARCH_TIMEOUT_SECONDS,
    RAG_SMART_AGENT_STRATEGY,
    RAG_TOP_K,
    REQUEST_TIMEOUT_SECONDS,
    TELEGRAM_MESSAGE_LIMIT,
)
from providers.main_client import main_client
from rag import index_store

from . import rag_compare_eval as ev
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


def run_mode(question_text: str, rag_enabled: bool, timeout: float | None = None) -> ev.ModeResult:
    """Ответ одного режима: свежий SmartAgent во временном каталоге (пустая память, профиль
    без полей, источники MCP выключены), слой rag включён или выключен. Настройки поиска —
    из config.py, как в рабочем режиме. Обращение к модели — клиентом без автоповторов и с
    таймаутом `timeout` (по умолчанию REQUEST_TIMEOUT_SECONDS). Сбой провайдера становится
    причиной в результате, а не исключением."""
    with tempfile.TemporaryDirectory(prefix="rag_compare_") as tmp:
        agent = SmartAgent(
            _EVAL_CHAT_ID,
            client=_fast_client,
            timeout=timeout if timeout is not None else REQUEST_TIMEOUT_SECONDS,
            memory_dir=tmp,
            mcp_moex_dir="",
            mcp_bybit_dir="",
            cbr_enabled=False,
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
            {"title": s.title, "chunk_index": s.chunk_index, "score": round(s.score, 3)}
            for s in answer.rag_sources
        ],
        search_failed=rag_context.FAILURE_WARNING in answer.warnings,
    )


def _judge_call(user_content: str, max_tokens: int, timeout: float) -> dict | None:
    """Один вызов оценщика. «thinking» отключён, как в research/temperature.py: оценщику нужен
    короткий JSON, а на моделях с рассуждениями весь лимит токенов уходил на скрытые
    размышления (finish_reason == "length", content пуст) — оценка терялась. Пустой content
    всё равно возможен (лимит) — его обрабатывает judge() повтором."""
    response = _fast_client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[
            {"role": "system", "content": RAG_COMPARE_JUDGE_SYSTEM_PROMPT},
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


def judge(
    question: ev.Question, answer: str, budget: QuestionBudget | None = None
) -> tuple[ev.Verdict | None, str | None]:
    """Оценка ответа по чек-листу фактов (вслепую по режиму). Один повтор с удвоенным лимитом
    токенов при обрезанном JSON или пустом ответе — только пока не исчерпан бюджет вопроса.
    Таймаут каждого вызова — остаток бюджета (без бюджета — REQUEST_TIMEOUT_SECONDS).
    Возвращает (оценка, None) или (None, причина)."""
    user_content = ev.build_judge_user_content(question.question, question.facts, answer)

    def call(max_tokens: int) -> dict | None:
        limit = budget.call_timeout() if budget else REQUEST_TIMEOUT_SECONDS
        # Жёсткий предел по часам вокруг вызова (таймаут клиента его не гарантирует).
        return ev.call_with_deadline(lambda: _judge_call(user_content, max_tokens, limit), limit)

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
    verdict = ev.parse_judge_response(data, len(question.facts))
    if verdict is None:
        return None, "оценщик вернул непригодный ответ"
    return verdict, None


def evaluate_question(
    question: ev.Question,
    index_titles: list[str],
    on_stage: Callable[[str], None] | None = None,
    budget_seconds: float = RAG_COMPARE_QUESTION_TIMEOUT_SECONDS,
    cancelled: Callable[[], bool] | None = None,
) -> ev.QuestionResult:
    """Оба режима по одному вопросу плюс оценка каждого полученного ответа, в рамках общего
    бюджета времени вопроса (budget_seconds): исчерпанный бюджет помечает оставшиеся этапы
    причиной «превышен лимит времени вопроса» без новых обращений. `on_stage` вызывается на
    входе в каждый этап (из рабочего потока) — по нему строится прогресс. `cancelled` — признак
    остановки прогона: поток, брошенный отменённой задачей, на следующем этапе завершается
    (RunCancelled) и не делает новых обращений к модели."""
    budget = QuestionBudget(budget_seconds)
    plan = (
        (ev.MODE_OFF, False, ev.STAGE_OFF_ANSWER, ev.STAGE_OFF_JUDGE),
        (ev.MODE_ON, True, ev.STAGE_ON_ANSWER, ev.STAGE_ON_JUDGE),
    )
    modes: dict[str, ev.ModeResult] = {}
    for mode_key, rag_enabled, answer_stage, judge_stage in plan:
        if cancelled and cancelled():
            raise RunCancelled
        if on_stage:
            on_stage(answer_stage)
        if budget.expired():
            mode = ev.ModeResult(error=ev.QUESTION_TIMEOUT_REASON)
        else:
            mode = run_mode(question.question, rag_enabled, budget.call_timeout())
        if mode.answer is not None:
            if cancelled and cancelled():
                raise RunCancelled
            if on_stage:
                on_stage(judge_stage)
            if budget.expired():
                mode.judge_error = ev.QUESTION_TIMEOUT_REASON
            else:
                mode.verdict, mode.judge_error = judge(question, mode.answer, budget)
        modes[mode_key] = mode
    return ev.QuestionResult(
        question=question,
        off=modes[ev.MODE_OFF],
        on=modes[ev.MODE_ON],
        missing_sources=ev.sources_not_indexed(question.sources, index_titles),
    )


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
        names = sorted(n for n in os.listdir(report_dir) if n.endswith(_REPORT_SUFFIX))
    except FileNotFoundError:
        return None
    if not names:
        return None
    path = os.path.join(report_dir, names[-1])
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


def _settings() -> dict:
    return {
        "strategy": RAG_SMART_AGENT_STRATEGY,
        "top_k": RAG_TOP_K,
        "min_score": RAG_MIN_SCORE,
        "search_timeout": RAG_SEARCH_TIMEOUT_SECONDS,
        "question_timeout": RAG_COMPARE_QUESTION_TIMEOUT_SECONDS,
        "max_consecutive_failures": RAG_COMPARE_MAX_CONSECUTIVE_FAILURES,
        "model": MAIN_MODEL,
        "embeddings_model": EMBEDDINGS_MODEL,
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


async def _run_comparison(
    bot,
    chat_id: int,
    progress_message: Message,
    questions: list[ev.Question],
    index_titles: list[str],
) -> None:
    """Фоновая задача прогона. Заканчивается одним из трёх способов: все вопросы обработаны,
    автоостановка (подряд проваленные вопросы) или остановка командой (отмена задачи) — в двух
    последних случаях сохраняется ЧАСТИЧНЫЙ отчёт. Флаг «прогон идёт» снимается в finally,
    поэтому после любой ошибки можно запустить заново."""
    global _run_in_progress, _run_task, _stop_reason, _finalizing, _progress_done
    loop = asyncio.get_running_loop()
    reporter = _ProgressReporter(loop, progress_message)
    started_at = _now()
    total = len(questions)
    results: list[ev.QuestionResult] = []
    stop_reason: str | None = None
    cancel_event = threading.Event()
    try:
        try:
            for index, question in enumerate(questions, start=1):
                reporter.post(ev.progress_text(index - 1, total, ev.STAGES[0], question.question))

                def on_stage(stage: str, done: int = index - 1, text: str = question.question):
                    reporter.post(ev.progress_text(done, total, stage, text))

                results.append(
                    await asyncio.to_thread(
                        evaluate_question,
                        question,
                        index_titles,
                        on_stage,
                        RAG_COMPARE_QUESTION_TIMEOUT_SECONDS,
                        cancel_event.is_set,
                    )
                )
                _progress_done = index
                if ev.should_stop(results, RAG_COMPARE_MAX_CONSECUTIVE_FAILURES):
                    limit = RAG_COMPARE_MAX_CONSECUTIVE_FAILURES
                    stop_reason = (
                        f"автоостановка: {_plural_questions(limit)} подряд завершились сбоем "
                        "(провайдер не отвечает или превышен лимит времени вопроса)"
                    )
                    break
        except asyncio.CancelledError:
            # Остановка командой: отмена доставлена один раз — снимаем её и доводим итог.
            asyncio.current_task().uncancel()
            cancel_event.set()  # брошенный рабочий поток не начнёт следующий этап
            stop_reason = _stop_reason or "по команде пользователя"

        _finalizing = True
        report = ev.build_report(
            started_at=started_at,
            finished_at=_now(),
            settings=_settings(),
            results=results,
            planned=total,
            index_missing=[p for r in results for p in r.missing_sources],
            status=ev.STATUS_STOPPED if stop_reason else ev.STATUS_COMPLETED,
            stop_reason=stop_reason,
        )
        await asyncio.to_thread(save_report, report)
        final_note = (
            f"⛔ Прогон остановлен: обработано {len(results)} из {total} вопросов."
            if stop_reason
            else ev.progress_text(total, total) + " Готово."
        )
        await reporter.finish(final_note)
        await _send_parts(bot, chat_id, ev.format_summary(report))
        await _send_parts(bot, chat_id, ev.format_table(report))
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
            "документов индекса). Формат и пример — в README. Другой путь задаётся переменной "
            "RAG_COMPARE_QUESTIONS_FILE."
        )
        return
    try:
        questions = ev.load_questions(QUESTIONS_PATH)
    except ev.QuestionSetError as exc:
        await update.message.reply_text(f"❌ Контрольный набор вопросов некорректен: {exc}.")
        return

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
        progress_message = await update.message.reply_text(
            ev.progress_text(0, len(questions))
            + "\nБот остаётся доступен; итог придёт сюда, подробности — "
            "/research_rag_compare_report <номер>, остановить — /research_rag_compare_stop."
        )
        task = asyncio.get_running_loop().create_task(
            _run_comparison(
                context.bot, update.effective_chat.id, progress_message, questions, index_titles
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
    if not context.args:
        text = ev.format_summary(report) + "\n\n" + ev.format_table(report)
    else:
        try:
            number = int(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "👉 Формат: /research_rag_compare_report или "
                "/research_rag_compare_report <номер вопроса>"
            )
            return
        text = ev.format_question_detail(report, number)
        if text is None:
            if 1 <= number <= planned:
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
