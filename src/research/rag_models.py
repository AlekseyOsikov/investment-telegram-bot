"""Команды /research_rag_models, /research_rag_models_stop и /research_rag_models_report —
сравнение моделей (в первую очередь локальной и облачной) на контрольных вопросах с включённым
слоем rag (design.md изменения add-rag-models-compare). Технический/исследовательский режим:
регистрируется и упоминается в /help только при RESEARCH_ENABLED, как остальные /research_*.

Правила и тексты (разбор настроек, выбор вопросов, метрики, агрегаты, отчёт) — в
research/rag_models_eval.py (чистый модуль с тестами); здесь — обращения к SmartAgent и судье,
фоновый прогон, отчёты на диске и обработчики Telegram. Правила оценки ответов, контрольный набор,
предел по часам, флаг единственного прогона и прогресс — из research/rag_compare.py.

Условия опыта: на каждый вопрос ОДИН общий поиск по индексу (эмбеддинг, кандидаты, отбор), его
материалы подаются всем моделям и всем повторам (`SmartAgent.ask(materials=…)`); каждый ответ строит
СВЕЖИЙ изолированный SmartAgent с клиентом и моделью пары (пустая память, переписывание вопроса
выключено, источники рыночных данных выключены). Порядок обращений: повтор → модели по очереди,
чтобы нагрузка на машину или провайдера не приходилась на одну модель. Судья вызывается после
каждого ответа строго последовательно (Kimi допускает один запрос одновременно).

Модели и судью задаёт оператор в .env (RAG_MODELS_*): это не рантайм-выбор модели пользователем.
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
from agents.smart_agent import SmartAgent
from config import (
    RAG_COMPARE_MAX_CONSECUTIVE_FAILURES,
    RAG_COMPARE_QUESTION_TIMEOUT_SECONDS,
    RAG_COMPARE_QUESTIONS_FILE,
    RAG_COMPARE_REPORT_DIR,
    RAG_INDEX_DIR,
    RAG_MODELS_COMPARE,
    RAG_MODELS_JUDGE,
    RAG_MODELS_QUESTIONS,
    RAG_MODELS_REPEATS,
    RAG_SEARCH_TIMEOUT_SECONDS,
    RAG_SMART_AGENT_STRATEGY,
    TELEGRAM_MESSAGE_LIMIT,
)
from providers.deepseek_client import DEEPSEEK_API_KEY, DEEPSEEK_MODEL_FLASH, deepseek_client
from providers.kimi_client import KIMI_API_KEY, KIMI_MODEL_K3, kimi_client
from providers.ollama_client import ollama_client
from rag import index_store

from . import rag_compare as rc
from . import rag_compare_eval as ev
from . import rag_models_eval as rme

logger = logging.getLogger(__name__)

_EVAL_CHAT_ID = "rag_models"
_EVAL_PROFILE = "eval"

_DEFAULT_MODELS = {"deepseek": DEEPSEEK_MODEL_FLASH, "kimi": KIMI_MODEL_K3}
_CLIENTS = {"ollama": ollama_client, "deepseek": deepseek_client, "kimi": kimi_client}
_API_KEYS = {"deepseek": bool(DEEPSEEK_API_KEY), "kimi": bool(KIMI_API_KEY)}

SEARCH_FAILED = ev.SEARCH_FAILED_REASON


# --------------------------------------------------------------------------- #
# Настройки и бэкенды
# --------------------------------------------------------------------------- #


def build_backend(spec: rme.ModelSpec) -> rc.LlmBackend:
    """Клиент пары без автоповторов SDK (быстрый отказ: зависший провайдер не умножает ожидание)
    и её модель."""
    client = _CLIENTS[spec.provider].with_options(max_retries=0)
    return rc.LlmBackend(client, spec.model, spec.provider)


def load_settings() -> tuple[list[rme.ModelSpec], rme.ModelSpec]:
    """Модели и судья из настроек оператора с проверкой ключей облачных провайдеров. Нарушение —
    rme.ModelSettingError с понятным сообщением (команда отвечает им и не начинает прогон)."""
    specs = rme.parse_model_specs(RAG_MODELS_COMPARE, _DEFAULT_MODELS)
    judge = rme.parse_judge_spec(RAG_MODELS_JUDGE, _DEFAULT_MODELS)
    missing = rme.missing_api_keys([*specs, judge], _API_KEYS)
    if missing:
        raise rme.ModelSettingError(
            "не задан ключ облачного провайдера, выбранного для сравнения или судьи: "
            + ", ".join(missing)
            + "."
        )
    return specs, judge


# --------------------------------------------------------------------------- #
# Вопрос: общий поиск, повторы, ответы и оценка (синхронно — через asyncio.to_thread)
# --------------------------------------------------------------------------- #


def _new_agent(tmp: str, name: str) -> SmartAgent:
    """Изолированный агент для общего поиска: пустая память, переписывание выключено, источники
    рыночных данных выключены."""
    agent = SmartAgent(
        f"{_EVAL_CHAT_ID}_{name}",
        client=rc._fast_client,  # noqa: SLF001 — клиент поиска не используется (нет переписывания)
        memory_dir=tmp,
        mcp_moex_dir="",
        mcp_bybit_dir="",
        cbr_enabled=False,
        rewrite_backend=None,
    )
    agent.create_profile(_EVAL_PROFILE, {})
    return agent


def search_once(
    question: ev.Question, budget: rc.QuestionBudget
) -> tuple[rag_context.Materials | None, list[dict], float, str | None]:
    """ОДИН поиск материалов на вопрос (design.md, решение 11). Возвращает (материалы, источники
    найденных фрагментов для отчёта, секунды, причина сбоя). Сбой — материалы None и причина."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rag_models_search_") as tmp:
        agent = _new_agent(tmp, "search")
        try:
            materials, _ = ev.call_with_deadline(
                lambda: agent.search_materials(question.question, None, []),
                min(budget.call_timeout(), RAG_SEARCH_TIMEOUT_SECONDS + 5),
            )
        except ev.DeadlineExceeded:
            return None, [], time.monotonic() - started, "поиск не уложился в предел времени"
        except Exception as exc:  # noqa: BLE001 — сбой поиска исключает вопрос, а не роняет прогон
            return None, [], time.monotonic() - started, rc._error_reason(exc)  # noqa: SLF001
        seconds = time.monotonic() - started
        if materials.warning:
            _, reason = agent.get_rag_status()
            return None, [], seconds, reason or SEARCH_FAILED
    sources = [
        {
            "title": c.title,
            "chunk_index": c.chunk_index,
            "score": round(c.score, 3),
            "chunk_id": c.chunk_id,
        }
        for c in materials.chunks
    ]
    return materials, sources, seconds, None


def answer_and_judge(
    question: ev.Question,
    backend: rc.LlmBackend,
    judge_backend: rc.LlmBackend,
    materials: rag_context.Materials,
    budget: rc.QuestionBudget,
    on_stage: Callable[[str], None] | None,
    cancelled: Callable[[], bool] | None,
) -> ev.ModeResult:
    """Один ответ модели по готовым материалам и его оценка судьёй в рамках бюджета вопроса.
    Сбой судьи не делает ответ сбоем генерации (judge_error)."""
    if cancelled and cancelled():
        raise rc.RunCancelled
    if on_stage:
        on_stage("ответ")
    if budget.expired():
        return ev.ModeResult(error=ev.QUESTION_TIMEOUT_REASON)
    mode = rc.run_mode(
        question.question, True, budget.call_timeout(), backend=backend, materials=materials
    )
    if mode.answer is None or mode.error:
        return mode
    if question.expect_abstain:
        return mode  # вопрос вне корпуса: зачёт — abstained, оценки по фактам нет
    if cancelled and cancelled():
        raise rc.RunCancelled
    if on_stage:
        on_stage("оценка судьёй")
    if budget.expired():
        mode.judge_error = ev.QUESTION_TIMEOUT_REASON
    else:
        mode.verdict, mode.judge_error = rc.judge(question, mode.answer, budget, judge_backend)
    if mode.citations:
        if cancelled and cancelled():
            raise rc.RunCancelled
        if on_stage:
            on_stage("оценка цитат судьёй")
        if budget.expired():
            mode.citation_judge_error = ev.QUESTION_TIMEOUT_REASON
        else:
            mode.citation_verdict, mode.citation_judge_error = rc.judge_citations(
                question,
                mode.answer,
                [c["quote"] for c in mode.citations],
                mode.fragments,
                budget,
                judge_backend,
            )
    return mode


def evaluate_question(
    question: ev.Question,
    models: list[tuple[rme.ModelSpec, rc.LlmBackend]],
    judge_backend: rc.LlmBackend,
    repeats: int,
    budget_seconds: float,
    cold_seen: set[str],
    on_stage: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> rme.QuestionRun:
    """Вопрос целиком: общий поиск, затем «повтор → модель → ответ → судья» последовательно.
    Сбой модели не прерывает остальные; сбой поиска исключает вопрос без обращений к моделям.
    `cold_seen` — модели, к которым в прогоне уже обращались (холодный старт — первое обращение;
    ответ «не знаю» без вызова модели прогрева не даёт и холодным не считается)."""
    budget = rc.QuestionBudget(budget_seconds)
    run = rme.QuestionRun(question=question)
    if cancelled and cancelled():
        raise rc.RunCancelled
    if on_stage:
        on_stage("поиск по индексу")
    materials, sources, seconds, error = search_once(question, budget)
    run.search_seconds = seconds
    if materials is None:
        run.search_error = error or SEARCH_FAILED
        return run
    run.search_sources = sources

    for repeat in range(1, repeats + 1):
        for spec, backend in models:

            def stage(text: str, spec: rme.ModelSpec = spec, repeat: int = repeat) -> None:
                if on_stage:
                    on_stage(f"повтор {repeat} из {repeats} · {spec.label} · {text}")

            cold = spec.key not in cold_seen
            result = answer_and_judge(
                question, backend, judge_backend, materials, budget, stage, cancelled
            )
            if result.llm_calls > 0 or result.error:
                cold_seen.add(spec.key)
            else:
                cold = False
            run.attempts.setdefault(spec.key, []).append(
                rme.Attempt(spec.key, repeat, result, cold=cold)
            )
    return run


# --------------------------------------------------------------------------- #
# Хранение отчётов
# --------------------------------------------------------------------------- #


def save_report(report: dict, report_dir: str = RAG_COMPARE_REPORT_DIR) -> str:
    """Пишет отчёт как `<UTC-время>_models.json` (временный файл + переименование). Возвращает
    путь файла."""
    os.makedirs(report_dir, exist_ok=True)
    name = rme.report_file_name(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    path = os.path.join(report_dir, name)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def load_latest_report(report_dir: str = RAG_COMPARE_REPORT_DIR) -> dict | None:
    """Последний отчёт сравнения моделей; None — отчётов ещё нет. Повреждённый файл или отчёт
    другого вида и версии — rc.ReportError."""
    try:
        name = rme.latest_models_report_name(os.listdir(report_dir))
    except FileNotFoundError:
        return None
    if name is None:
        return None
    try:
        with open(os.path.join(report_dir, name), encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise rc.ReportError(str(exc)) from exc
    if not rme.is_models_report(report):
        raise rc.ReportError("отчёт другого вида или версии формата")
    return report


# --------------------------------------------------------------------------- #
# Фоновый прогон
# --------------------------------------------------------------------------- #


def _settings(models: list[rme.ModelSpec], judge: rme.ModelSpec, repeats: int) -> dict:
    """Настройки прогона для отчёта: рабочие настройки поиска; модель основного потока и
    переписывание к сравнению не относятся (переписывание выключено, модели — из пар)."""
    settings = rc.run_settings()
    for key in ("level", "model", "rewrite_provider", "rewrite_model", "rewrite_search_mode"):
        settings.pop(key, None)
    settings.update(
        {
            "rewrite": "off",
            "repeats": repeats,
            "models": [m.key for m in models],
            "judge": judge.key,
        }
    )
    return settings


def _progress_text(done: int, total: int, index: int, stage: str, question: str) -> str:
    return (
        f"⏳ Сравнение моделей: вопрос {index} из {total} · {stage}\n"
        f"Обработано вопросов: {done}.\n{question}"
    )


async def _run_comparison(
    bot,
    chat_id: int,
    progress_message: Message,
    questions: list[ev.Question],
    models: list[tuple[rme.ModelSpec, rc.LlmBackend]],
    judge: tuple[rme.ModelSpec, rc.LlmBackend],
    repeats: int,
) -> None:
    """Фоновая задача прогона. Заканчивается всеми вопросами, автоостановкой или остановкой
    командой (в двух последних случаях сохраняется ЧАСТИЧНЫЙ отчёт). Флаг «прогон идёт»
    снимается в finally."""
    loop = asyncio.get_running_loop()
    reporter = rc.ProgressReporter(loop, progress_message)
    started_at = rc.now_utc()
    total = len(questions)
    runs: list[rme.QuestionRun] = []
    stop_reason: str | None = None
    cancel_event = threading.Event()
    cold_seen: set[str] = set()
    budget_seconds = rme.question_budget_seconds(
        RAG_COMPARE_QUESTION_TIMEOUT_SECONDS, len(models), repeats
    )
    specs = [spec for spec, _ in models]
    try:
        try:
            for index, question in enumerate(questions, start=1):

                def on_stage(
                    stage: str,
                    done: int = index - 1,
                    number: int = index,
                    text: str = question.question,
                ) -> None:
                    reporter.post(_progress_text(done, total, number, stage, text))

                on_stage("подготовка")
                runs.append(
                    await asyncio.to_thread(
                        evaluate_question,
                        question,
                        models,
                        judge[1],
                        repeats,
                        budget_seconds,
                        cold_seen,
                        on_stage,
                        cancel_event.is_set,
                    )
                )
                rc.set_progress_done(index)
                limit = RAG_COMPARE_MAX_CONSECUTIVE_FAILURES
                if rme.should_stop(runs, limit):
                    stop_reason = (
                        f"автоостановка: {rc.plural_questions(limit)} подряд завершились сбоем "
                        "(все обращения к моделям провалились или поиск недоступен)"
                    )
                    break
        except asyncio.CancelledError:
            # Остановка командой: отмена доставлена один раз — снимаем её и доводим итог.
            asyncio.current_task().uncancel()
            cancel_event.set()  # брошенный рабочий поток не начнёт следующий этап
            stop_reason = rc.current_stop_reason() or "по команде пользователя"

        rc.set_finalizing()
        status = ev.STATUS_STOPPED if stop_reason else ev.STATUS_COMPLETED
        report = rme.build_report(
            started_at=started_at,
            finished_at=rc.now_utc(),
            settings=_settings(specs, judge[0], repeats),
            models=specs,
            judge=judge[0],
            runs=runs,
            planned=total,
            repeats=repeats,
            status=status,
            stop_reason=stop_reason,
        )
        await asyncio.to_thread(save_report, report)
        if stop_reason:
            final_note = f"⛔ Прогон остановлен: обработано {len(runs)} из {total} вопросов."
        else:
            final_note = f"⏳ Сравнение моделей: обработано {total} из {total} вопросов. Готово."
        await reporter.finish(final_note)
        await rc.send_parts(bot, chat_id, rme.format_summary(report))
    except Exception:  # noqa: BLE001 — сообщаем в чат и освобождаем флаг (finally)
        logger.exception("Прогон сравнения моделей завершился неожиданной ошибкой.")
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Прогон сравнения моделей прервался из-за неожиданной ошибки. Результат "
                "не сохранён. Можно запустить /research_rag_models заново.",
            )
        except TelegramError:
            logger.exception("Не удалось сообщить о сбое прогона в чат.")
    finally:
        reporter.closed = True
        rc.release_run()


# --------------------------------------------------------------------------- #
# Обработчики
# --------------------------------------------------------------------------- #


async def rag_models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/research_rag_models — проверяет настройки, набор и индекс, запускает фоновый прогон и
    сразу отвечает; итог придёт отдельным сообщением."""
    # Все проверки до занятия флага синхронные (без await между проверкой и занятием).
    if not RAG_MODELS_COMPARE and not RAG_MODELS_JUDGE:
        await update.message.reply_text(
            "📭 Сравнение моделей не настроено. Оператору нужно задать в .env "
            "RAG_MODELS_COMPARE (пары «провайдер:модель», не меньше двух) и RAG_MODELS_JUDGE "
            "(модель-судья) и перезапустить бота — подробности в .env.example."
        )
        return
    try:
        specs, judge_spec = load_settings()
    except rme.ModelSettingError as exc:
        await update.message.reply_text(f"❌ Настройки сравнения моделей: {exc}")
        return

    if not os.path.isfile(RAG_COMPARE_QUESTIONS_FILE):
        await update.message.reply_text(
            f"📭 Контрольный набор вопросов не найден: {RAG_COMPARE_QUESTIONS_FILE}.\n"
            "Он тот же, что у /research_rag_compare: создайте файл (формат — в README) или "
            "задайте другой путь переменной RAG_COMPARE_QUESTIONS_FILE."
        )
        return
    try:
        all_questions = ev.load_questions(RAG_COMPARE_QUESTIONS_FILE)
    except ev.QuestionSetError as exc:
        await update.message.reply_text(f"❌ Контрольный набор вопросов некорректен: {exc}.")
        return
    selection = rme.select_questions(all_questions, RAG_MODELS_QUESTIONS)
    if selection.error:
        notes = "\n".join(selection.notes)
        await update.message.reply_text(
            f"📭 Прогон не начат: {selection.error}" + (f"\n{notes}" if notes else "")
        )
        return
    questions = list(selection.questions)

    if rc.is_run_in_progress():
        done, total = rc.run_progress()
        await update.message.reply_text(
            f"⏳ Прогон уже идёт: обработано {done} из {total} вопросов.\n"
            "Итог придёт в чат, который его запустил. Остановить — /research_rag_models_stop "
            "(или /research_rag_compare_stop для сравнения «без RAG / с RAG»)."
        )
        return

    if not index_store.index_exists(RAG_SMART_AGENT_STRATEGY, RAG_INDEX_DIR):
        await update.message.reply_text(
            f"📭 Индекс стратегии «{RAG_SMART_AGENT_STRATEGY}» не построен — сравнение с RAG без "
            "индекса ничего не измерило бы. Сначала выполните `make index` на хосте бота."
        )
        return

    models = [(spec, build_backend(spec)) for spec in specs]
    judge = (judge_spec, build_backend(judge_spec))
    if not rc.acquire_run(len(questions)):  # защита: флаг заняли между проверкой и сейчас
        await update.message.reply_text("⏳ Прогон уже идёт.")
        return
    try:
        for note in selection.notes:
            await update.message.reply_text(f"ℹ️ {note}")
        attempts = len(questions) * len(models) * RAG_MODELS_REPEATS
        progress_message = await update.message.reply_text(
            f"⏳ Сравнение моделей: {', '.join(s.label for s in specs)}.\n"
            f"Вопросов: {len(questions)}, повторов: {RAG_MODELS_REPEATS}, ответов ≈ {attempts}, "
            f"каждый оценивает судья {judge_spec.label}.\n"
            f"Судье уходят вопросы, ответы и тексты цитируемых фрагментов корпуса"
            + (
                " (облачный провайдер)."
                if judge_spec.provider != "ollama"
                else " (локальный сервер Ollama)."
            )
            + "\nБот остаётся доступен; итог придёт сюда, подробности — "
            "/research_rag_models_report <номер>, остановить — /research_rag_models_stop."
        )
        task = asyncio.get_running_loop().create_task(
            _run_comparison(
                context.bot,
                update.effective_chat.id,
                progress_message,
                questions,
                models,
                judge,
                RAG_MODELS_REPEATS,
            )
        )
    except BaseException:
        rc.release_run()
        raise
    rc.set_run_task(task)
    rc.track_background_task(task)


async def rag_models_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/research_rag_models_stop — останавливает идущий прогон (общий флаг с /research_rag_compare):
    частичный итог придёт в чат, запустивший прогон."""
    await rc.reply_stop_result(update, rc.request_stop())


async def rag_models_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/research_rag_models_report [номер] — сводка последнего отчёта сравнения моделей или детали
    вопроса со всеми повторами рядом."""
    try:
        report = await asyncio.to_thread(load_latest_report)
    except rc.ReportError:
        logger.exception("Не удалось прочитать последний отчёт сравнения моделей.")
        await update.message.reply_text(
            "⚠️ Не удалось прочитать последний отчёт сравнения моделей: файл повреждён или "
            "другого формата. Запустите прогон заново: /research_rag_models."
        )
        return
    if report is None:
        await update.message.reply_text(
            "📭 Отчётов сравнения моделей пока нет. Запустите прогон: /research_rag_models."
        )
        return

    processed = len(report["questions"])
    planned = report.get("planned", processed)
    if not context.args:
        text = rme.format_summary(report)
    else:
        try:
            number = int(context.args[0])
        except ValueError:
            await update.message.reply_text(
                "👉 Формат: /research_rag_models_report или "
                "/research_rag_models_report <номер вопроса>"
            )
            return
        text = rme.format_question_detail(report, number)
        if text is None:
            # Номера — это номера вопросов контрольного набора (в прогоне их может быть
            # «1, 2, 3, 4, 9»), а не порядок в прогоне.
            available = rme.question_numbers(report)
            listing = ", ".join(str(n) for n in available) or "нет"
            note = (
                f" Прогон остановлен: обработано {processed} из {planned} вопросов."
                if report.get("status") == ev.STATUS_STOPPED
                else ""
            )
            await update.message.reply_text(
                f"Вопроса с номером {number} в этом отчёте нет.{note}\n"
                f"Доступные номера (из контрольного набора): {listing}."
            )
            return

    for part in ev.split_text(text, TELEGRAM_MESSAGE_LIMIT):
        await update.message.reply_text(part)


def build_rag_models_handlers() -> list[CommandHandler]:
    private = filters.ChatType.PRIVATE
    return [
        CommandHandler("research_rag_models", rag_models_command, filters=private),
        CommandHandler("research_rag_models_stop", rag_models_stop_command, filters=private),
        CommandHandler("research_rag_models_report", rag_models_report_command, filters=private),
    ]

