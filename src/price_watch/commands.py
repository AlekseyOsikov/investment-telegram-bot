"""Команды опроса цен: /watch, /watch_status, /watch_report, /watch_stop.

Модель в этом потоке НЕ участвует: команды сами вызывают инструменты расписания
сервера mcp-moex (mcp_integration/watch_session.py), а `chat_id` берут из
`update.effective_chat.id`, а не из аргументов команды. Серверу уходят только
идентификатор чата, тикеры и периоды — не текст остальных сообщений, не история и не
память агентов (см. требование «Границы данных и хранения» в спеке price-watch).

Регистрируются в main.py только при активной возможности (config.PRICE_WATCH_ACTIVE)
и только в личных чатах. Ошибки обращения к серверу переводятся в сообщения на
русском отдельными except по тому же паттерну, что mcp_integration/tools_command.py:
FileNotFoundError (нет `uv`), TimeoutError, ошибка самого инструмента (WatchToolError,
текст сервера) и общий fallback с logger.exception. Сбой сервера не влияет на остальные
команды и сообщения бота.

Плановая выдача сводок — не здесь, а в price_watch/scheduler.py; связь с ним — только
`schedule_at` после успешной /watch (планировщик лежит в `application.bot_data`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import CommandHandler, ContextTypes, filters

from mcp_integration.watch_session import WatchToolError, call_watch_tool

from .parse import ParseError, parse_watch_args
from .scheduler import SCHEDULER_KEY, WatchScheduler, split_message
from .texts import (
    NO_WATCH_TEXT,
    USAGE_TEXT,
    friendly_tool_error,
    watch_set_text,
    watch_status_text,
    watch_stop_text,
)

logger = logging.getLogger(__name__)


async def _call_or_report(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    name: str,
    arguments: dict,
    *,
    failure: str,
    report_tool_error: bool = True,
) -> dict | None:
    """Вызывает инструмент сервера; при сбое отвечает пользователю и возвращает None.

    `failure` — начало сообщения об отказе сервера («Не удалось задать опрос»); дальше
    идёт его текст как есть. При report_tool_error=False отказ инструмента не
    показывается, а поднимается заново — вызывающий код решает сам (см. /watch_report).
    """
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        return await call_watch_tool(name, arguments)
    except WatchToolError as error:
        if not report_tool_error:
            raise
        logger.warning("Сервер расписания отклонил вызов %s: %s", name, error)
        await update.message.reply_text(f"⚠️ {failure}: {friendly_tool_error(str(error))}")
    except FileNotFoundError:
        logger.error("Команда запуска сервера расписания не найдена на хосте.")
        await update.message.reply_text(
            "❌ Не удалось запустить сервер данных биржи: программа запуска (uv) не найдена "
            "на хосте бота. Администратору нужно проверить окружение."
        )
    except TimeoutError:
        logger.error("Сервер расписания не ответил за отведённое время (%s).", name)
        await update.message.reply_text(
            "⏱ Сервер данных биржи не ответил вовремя. Попробуй повторить команду позже."
        )
    except Exception:  # noqa: BLE001 — последний рубеж для протокольных и прочих ошибок сервера
        logger.exception("Не удалось выполнить вызов %s сервера расписания.", name)
        await update.message.reply_text(
            "❌ Не удалось соединиться с сервером данных биржи. Попробуй повторить команду "
            "позже; подробности записаны в журнал."
        )
    return None


def _earliest(*moments: str | None) -> datetime | None:
    """Самый ранний из ISO-моментов, что удалось разобрать."""
    parsed: list[datetime] = []
    for value in moments:
        if not value:
            continue
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            continue
        parsed.append(moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc))
    return min(parsed) if parsed else None


async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/watch <тикеры> <период опроса> <период сводки> — задать или заменить опрос."""
    request = parse_watch_args(list(context.args or []))
    if isinstance(request, ParseError):
        text = USAGE_TEXT if request.reason is None else f"⚠️ {request.reason}\n\n{USAGE_TEXT}"
        await update.message.reply_text(text)
        return

    result = await _call_or_report(
        update,
        context,
        "watch_set",
        {
            "chat_id": update.effective_chat.id,
            "secids": request.secids,
            "poll_interval": request.poll_interval,
            "report_interval": request.report_interval,
        },
        failure="Не удалось задать опрос",
    )
    if result is None:
        return

    # Сначала планировщик, потом ответ: сбой отправки ответа не должен оставить новый
    # опрос без пробуждения.
    scheduler: WatchScheduler | None = context.application.bot_data.get(SCHEDULER_KEY)
    due = _earliest(result.get("next_poll_at"), result.get("next_report_at"))
    if scheduler is not None and due is not None:
        scheduler.schedule_at(due)

    await update.message.reply_text(watch_set_text(result))


async def watch_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/watch_status — состояние опроса этого чата."""
    result = await _call_or_report(
        update,
        context,
        "watch_status",
        {"chat_id": update.effective_chat.id},
        failure="Не удалось получить состояние опроса",
    )
    if result is not None:
        await update.message.reply_text(watch_status_text(result))


async def watch_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/watch_report — сводка по накопленному прямо сейчас, без сдвига расписания."""
    chat_id = update.effective_chat.id
    try:
        result = await _call_or_report(
            update,
            context,
            "watch_get_report",
            {"chat_id": chat_id},
            failure="Не удалось получить сводку",
            report_tool_error=False,
        )
    except WatchToolError as error:
        # Отказ «опрос не задан» сервер пишет для модели (ссылается на инструмент
        # watch_set), пользователю он не годится. Уточняем состоянием и говорим своими
        # словами; любой другой отказ показываем как есть.
        status = await _call_or_report(
            update, context, "watch_status", {"chat_id": chat_id},
            failure="Не удалось получить состояние опроса",
        )
        if status is None:
            return
        if not status.get("active"):
            await update.message.reply_text(NO_WATCH_TEXT)
        else:
            await update.message.reply_text(
                f"⚠️ Не удалось получить сводку: {friendly_tool_error(str(error))}"
            )
        return
    if result is None:
        return
    for chunk in split_message(result.get("text") or ""):
        if chunk.strip():
            await update.message.reply_text(chunk)


async def watch_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/watch_stop — остановить опрос этого чата и удалить его замеры."""
    result = await _call_or_report(
        update,
        context,
        "watch_stop",
        {"chat_id": update.effective_chat.id},
        failure="Не удалось остановить опрос",
    )
    if result is not None:
        await update.message.reply_text(watch_stop_text(result))


def build_price_watch_handlers() -> list[CommandHandler]:
    """Четыре команды опроса цен, только для личных чатов."""
    private = filters.ChatType.PRIVATE
    return [
        CommandHandler("watch", watch_command, filters=private),
        CommandHandler("watch_status", watch_status_command, filters=private),
        CommandHandler("watch_report", watch_report_command, filters=private),
        CommandHandler("watch_stop", watch_stop_command, filters=private),
    ]
