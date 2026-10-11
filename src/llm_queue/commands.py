"""Команды очереди (`/queue`, `/queue_cancel`), хуки жизненного цикла шлюза и постановка задач.

`/queue` и `/queue_cancel` — личные чаты, не под флагом RESEARCH: это действия пользователя
над СОБСТВЕННЫМИ вопросами, а не исследовательские команды. Номер вопроса — его место в
списке чата (с 1, в порядке отправки); чужие вопросы по номеру недостижимы, потому что
выбор идёт только среди вопросов чата отправителя.

`/cancel` НЕ занимается: он остаётся выходом из режимов (agents/*_command.py), а выход
снимает вопросы чата через `cancel_chat_questions`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from telegram import Message, Update
from telegram.ext import Application, CommandHandler, ContextTypes, filters

from config import (
    LLM_MAX_CONCURRENCY,
    LLM_MAX_PENDING_PER_USER,
    LLM_QUEUE_MAX,
    LLM_QUEUE_TTL_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)

from . import rules
from .gate import GATE_KEY, JobContext, LlmGate

logger = logging.getLogger(__name__)


def get_gate(application: Application) -> LlmGate:
    """Шлюз приложения (создаётся в post_init)."""
    return application.bot_data[GATE_KEY]


async def start_gate(application: Application) -> None:
    """post_init приложения: создаёт шлюз и кладёт его в bot_data."""
    application.bot_data[GATE_KEY] = LlmGate(
        concurrency=LLM_MAX_CONCURRENCY,
        max_pending_per_user=LLM_MAX_PENDING_PER_USER,
        queue_max=LLM_QUEUE_MAX,
        ttl_seconds=LLM_QUEUE_TTL_SECONDS,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )
    logger.info(
        "Очередь LLM-задач: параллелизм %d, вопросов на чат %d, глубина %d, срок ожидания %d с.",
        LLM_MAX_CONCURRENCY,
        LLM_MAX_PENDING_PER_USER,
        LLM_QUEUE_MAX,
        LLM_QUEUE_TTL_SECONDS,
    )


async def stop_gate(application: Application) -> None:
    """post_stop приложения: отменяет все задачи очереди."""
    gate = application.bot_data.pop(GATE_KEY, None)
    if gate is not None:
        await gate.shutdown()


async def enqueue_message(
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    kind: str,
    text: str,
    run: Callable[[JobContext], Awaitable[None]],
    *,
    priority: int = rules.INTERACTIVE,
) -> bool:
    """Ставит вопрос чата сообщения в очередь и сообщает статус, если слот сразу не достаётся.

    Возвращает False, если вопрос НЕ принят (лимит чата или «бот занят») — пользователь уже
    получил сообщение; вызывающий просто возвращает своё состояние диалога.
    """
    gate = get_gate(context.application)
    chat_id = message.chat_id
    bot = context.bot

    async def notify(note: str) -> None:
        await bot.send_message(chat_id=chat_id, text=note)

    submission = gate.submit(chat_id, kind, text, run, priority=priority, notify=notify)
    if submission.status == "chat_limit":
        await message.reply_text(rules.text_chat_limit(submission.position, submission.estimate))
        return False
    if submission.status == "busy":
        await message.reply_text(rules.text_busy())
        return False
    if submission.queued and submission.estimate is not None:
        await message.reply_text(rules.text_accepted(submission.position, submission.estimate))
    return True


async def enqueue_job(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    kind: str,
    text: str,
    run: Callable[[JobContext], Awaitable[None]],
    *,
    priority: int = rules.INTERACTIVE,
) -> bool:
    """`enqueue_message` для обработчика с `update` (сообщение пользователя)."""
    return await enqueue_message(update.message, context, kind, text, run, priority=priority)


def cancel_chat_questions(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, reason: str = "exit"
) -> int:
    """Снимает ожидающие и выполняемые вопросы чата (выход из режима). Возвращает их число."""
    gate = context.application.bot_data.get(GATE_KEY)
    if gate is None:
        return 0
    return len(gate.cancel_chat(chat_id, reason))


def saved_notice(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str | None = None,
    was_saved: Callable[[Any], bool] | None = None,
):
    """Колбэк `late` для `JobContext.run_thread`: вопрос сняли, а поток всё же дописал обмен.

    Если поток завершился без исключения (обмен записан), пользователю сообщается, что ответ
    не отправляется, но обмен уже сохранён; если `AskCancelled` или другая ошибка —
    молчание (ничего не записано). `was_saved(result)` уточняет, что считать записью, когда
    результат — составной (например, три ответа `/agent_compare`).
    """
    bot = context.bot

    async def late(result, error) -> None:
        saved = result is not None if was_saved is None else was_saved(result)
        if error is None and saved:
            await bot.send_message(chat_id=chat_id, text=text or rules.text_already_saved())

    return late


async def queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/queue — свои вопросы в очереди с позицией и оценкой."""
    gate = get_gate(context.application)
    items = gate.view_chat(update.effective_chat.id)
    await update.message.reply_text(rules.format_queue_list(items))


async def queue_cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/queue_cancel [номер] — снять все свои вопросы или один по номеру из /queue."""
    gate = get_gate(context.application)
    chat_id = update.effective_chat.id
    arg = rules.parse_cancel_arg(context.args or [])
    if arg.error:
        await update.message.reply_text(rules.text_bad_number())
        return
    jobs = gate.chat_jobs(chat_id)
    if not jobs:
        await update.message.reply_text(rules.text_nothing_to_cancel())
        return
    if arg.all_jobs:
        cancelled = gate.cancel_chat(chat_id, "user")
        await update.message.reply_text(rules.text_cancel_all(len(cancelled)))
        return
    job = rules.select_by_number(jobs, arg.number)
    if job is None:
        await update.message.reply_text(rules.text_bad_number())
        return
    was_running = gate.cancel_job(job, "user")
    await update.message.reply_text(rules.text_cancel_one(arg.number, was_running))


def build_queue_handlers() -> list[CommandHandler]:
    """Две команды очереди, только для личных чатов."""
    private = filters.ChatType.PRIVATE
    return [
        CommandHandler("queue", queue_command, filters=private),
        CommandHandler("queue_cancel", queue_cancel_command, filters=private),
    ]
