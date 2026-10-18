"""Команды дайджеста новостей: /news, /news_stop, /news_status (только личные чаты).

Модель в командах НЕ участвует: они вызывают инструменты сервера новостей
(news_digest/news_session.py) и опроса цен (mcp_integration/watch_session.py), а `chat_id` берут из
`update.effective_chat.id`, а не из аргументов команды. Серверу уходят только идентификатор чата и
тикеры — не текст остальных сообщений, не история и не память агентов (спека news-digest,
требование «Хранение только публичных данных»). Тикеры отдельно не вводятся: список бумаг всегда из
текущего опроса `/watch` этого чата.

Регистрируются в main.py только при NEWS_ACTIVE. Ошибки обращения к серверам переводятся в
сообщения на русском отдельными except (как price_watch/commands.py). Плановая работа — в
news_digest/scheduler.py; связь с ней — только `request_tick` после успешной `/news`.
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import CommandHandler, ContextTypes, filters

from config import (
    NEWS_DIGEST_INTERVAL_HOURS,
    NEWS_MAX_MESSAGES_PER_DAY,
)
from mcp_integration.watch_session import WatchToolError, call_watch_tool

from . import rules, texts
from .news_session import NewsToolError, call_news_tool
from .scheduler import SCHEDULER_KEY, NewsScheduler

logger = logging.getLogger(__name__)


async def _call(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    caller,
    name: str,
    arguments: dict,
    failure: str,
) -> dict | None:
    """Вызывает инструмент; при сбое отвечает пользователю и возвращает None."""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        return await caller(name, arguments)
    except (NewsToolError, WatchToolError) as error:
        logger.warning("Сервер отклонил вызов %s: %s", name, error)
        await update.message.reply_text(f"⚠️ {failure}: {error}")
    except FileNotFoundError:
        logger.error("Команда запуска сервера не найдена на хосте (%s).", name)
        await update.message.reply_text(
            f"❌ {failure}: на хосте бота не найдена программа запуска сервера. "
            "Администратору нужно проверить окружение."
        )
    except TimeoutError:
        logger.error("Сервер не ответил за отведённое время (%s).", name)
        await update.message.reply_text(
            f"⏱ {failure}: сервер не ответил вовремя. Попробуй повторить команду позже."
        )
    except Exception:  # noqa: BLE001 — последний рубеж для протокольных и прочих ошибок сервера
        logger.exception("Не удалось выполнить вызов %s.", name)
        await update.message.reply_text(texts.NEWS_FAILED_TEXT)
    return None


async def _watched_tickers(update, context) -> list[str] | None:
    """Тикеры из опроса цен чата; [] — опроса нет; None — сбой (ответ уже отправлен)."""
    status = await _call(
        update,
        context,
        call_watch_tool,
        "watch_status",
        {"chat_id": update.effective_chat.id},
        "Не удалось получить опрос цен",
    )
    if status is None:
        return None
    return [str(s).upper() for s in status.get("secids") or []] if status.get("active") else []


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/news — включить дайджест по бумагам из /watch этого чата."""
    tickers = await _watched_tickers(update, context)
    if tickers is None:
        return
    if not tickers:
        await update.message.reply_text(texts.NEED_WATCH_TEXT)
        return

    chat_id = update.effective_chat.id
    failure = "Не удалось включить дайджест новостей"
    if (
        await _call(update, context, call_news_tool, "news_enable", {"chat_id": chat_id}, failure)
        is None
    ):
        return
    coverage = await _call(
        update, context, call_news_tool, "news_coverage", {"tickers": tickers}, failure
    )
    if coverage is None:
        return

    # Сначала планировщик, потом ответ: сбой отправки ответа не должен оставить включённый
    # дайджест без пробуждения.
    scheduler: NewsScheduler | None = context.application.bot_data.get(SCHEDULER_KEY)
    if scheduler is not None:
        scheduler.request_tick()
    await update.message.reply_text(
        texts.news_enabled_text(
            tickers,
            NEWS_DIGEST_INTERVAL_HOURS,
            NEWS_MAX_MESSAGES_PER_DAY,
            coverage.get("uncovered") or [],
        )
    )


async def news_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/news_status — состояние дайджеста этого чата."""
    chat_id = update.effective_chat.id
    failure = "Не удалось получить состояние дайджеста"
    state = await _call(
        update, context, call_news_tool, "news_state", {"chat_id": chat_id}, failure
    )
    if state is None:
        return
    if not state.get("enabled"):
        await update.message.reply_text(
            texts.news_status_text(False, [], None, 0, NEWS_MAX_MESSAGES_PER_DAY)
        )
        return

    tickers = await _watched_tickers(update, context)
    if tickers is None:
        return
    uncovered: list[str] = []
    if tickers:
        coverage = await _call(
            update, context, call_news_tool, "news_coverage", {"tickers": tickers}, failure
        )
        if coverage is None:
            return
        uncovered = coverage.get("uncovered") or []

    enabled_at = rules.parse_moment(state.get("enabled_at"))
    next_digest = (
        rules.next_digest_at(
            enabled_at, rules.parse_moment(state.get("last_digest_at")), NEWS_DIGEST_INTERVAL_HOURS
        ).isoformat()
        if enabled_at
        else None
    )
    await update.message.reply_text(
        texts.news_status_text(
            True,
            tickers,
            next_digest,
            int(state.get("messages_today") or 0),
            NEWS_MAX_MESSAGES_PER_DAY,
            uncovered,
        )
    )


async def news_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/news_stop — выключить дайджест этого чата и удалить его отметки."""
    result = await _call(
        update,
        context,
        call_news_tool,
        "news_disable",
        {"chat_id": update.effective_chat.id},
        "Не удалось выключить дайджест новостей",
    )
    if result is not None:
        await update.message.reply_text(texts.news_stop_text(bool(result.get("was_enabled"))))


def build_news_digest_handlers() -> list[CommandHandler]:
    """Три команды дайджеста новостей, только для личных чатов."""
    private = filters.ChatType.PRIVATE
    return [
        CommandHandler("news", news_command, filters=private),
        CommandHandler("news_status", news_status_command, filters=private),
        CommandHandler("news_stop", news_stop_command, filters=private),
    ]
