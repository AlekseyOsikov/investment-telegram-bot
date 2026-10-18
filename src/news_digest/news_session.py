"""Обращения к инструментам сервера новостей (`mcp_servers/news/`): один процесс на обращение.

Механика та же, что у `mcp_integration/watch_session.py`: «запустить процесс → рукопожатие → один
вызов → закрыть процесс», без shell, тайм-аут на весь цикл, развёртка `ExceptionGroup` из одного
исключения. Процесс не живёт между обращениями, пока бот разбирает новости моделью и отправляет
сообщения в Telegram. Сервер запускается ТЕМ ЖЕ интерпретатором, что и бот; путь к БД абсолютный,
ленты и срок хранения — настройки оператора (в схемах инструментов их нет). Инструменты сервера
изменяют хранилище, поэтому модели не отдаются ни одним слоем бота.

Исключения запуска и обмена (FileNotFoundError, TimeoutError, ошибки протокола) НЕ перехватываются —
их переводит вызывающий код (`friendly_error`). Отказ самого инструмента — `NewsToolError` с
русским текстом сервера. Чистые функции — tests/test_news_session.py.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import NEWS_DB, NEWS_FEEDS, NEWS_RETENTION_DAYS
from mcp_integration.watch_session import (
    WatchProtocolError,
    WatchToolError,
    result_to_dict,
    unwrap_exception_group,
)

SERVER_SCRIPT = str(Path(__file__).resolve().parents[2] / "mcp_servers" / "news" / "news_server.py")
# Сбор опрашивает все ленты (до ~15 с на каждую) и разбирает ответы: запас больше, чем у /watch.
SERVER_CALL_TIMEOUT_SECONDS = 180.0


class NewsToolError(Exception):
    """Сервер выполнил вызов и сообщил об ошибке (isError). `str(error)` — русский текст сервера."""


def build_news_server_params(
    db_path: str, feeds: Sequence[str], retention_days: int
) -> StdioServerParameters:
    """Параметры запуска сервера новостей: тот же интерпретатор, список аргументов без shell
    (значения не могут внедрить команду), путь к БД — абсолютный."""
    args = [SERVER_SCRIPT, "--db", db_path, "--retention-days", str(retention_days)]
    if feeds:
        args += ["--feeds", *feeds]
    return StdioServerParameters(command=sys.executable, args=args)


def friendly_error(error: BaseException) -> str:
    """Русское объяснение сбоя обращения к серверу новостей (без технических подробностей)."""
    if isinstance(error, NewsToolError):
        return str(error)
    if isinstance(error, FileNotFoundError):
        return "не найдена программа запуска сервера новостей на хосте бота"
    if isinstance(error, TimeoutError):
        return "сервер новостей не ответил вовремя"
    if isinstance(error, WatchProtocolError):
        return "сервер новостей вернул неожиданный ответ"
    return "не удалось соединиться с сервером новостей"


async def call_news_tool(
    name: str,
    arguments: dict | None = None,
    *,
    db_path: str = NEWS_DB,
    feeds: Sequence[str] = tuple(NEWS_FEEDS),
    retention_days: int = NEWS_RETENTION_DAYS,
    timeout: float = SERVER_CALL_TIMEOUT_SECONDS,
) -> dict:
    """Вызывает один инструмент сервера новостей и возвращает результат словарём."""
    params = build_news_server_params(db_path, feeds, retention_days)
    try:
        async with asyncio.timeout(timeout):
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments or {})
    except Exception as error:  # noqa: BLE001 — разворачиваем группу и поднимаем заново
        leaf = unwrap_exception_group(error)
        if leaf is error:
            raise
        raise leaf from None

    if not hasattr(result, "content"):
        raise WatchProtocolError(f"Сервер новостей вернул неожиданный ответ на вызов {name}.")
    try:
        return result_to_dict(result)
    except WatchToolError as error:
        raise NewsToolError(str(error)) from None
