"""Обращения к инструментам расписания сервера рыночных данных (mcp-moex, режим `--watch-db`).

Здесь только механика MCP, без Telegram и без знания о том, как устроены команды и
планировщик (price_watch/) — той же границей, что у market_session.py. В отличие от
него, сессия живёт на ОДНО обращение: «запустить процесс → рукопожатие → один вызов
инструмента → закрыть процесс». Планировщик делает два обращения за тик (выдача
сводок, затем подтверждение), а не держит сессию открытой, пока бот отправляет
сообщения в Telegram: процесс сервера не должен жить между обращениями (см. требование
«Освобождение ресурсов после запроса» в спеке mcp-integration).

Этот сервер отдельный от сервера, который получает МОДЕЛЬ (market_session.py):
инструменты расписания принимают chat_id от клиента и изменяют хранилище, поэтому
модель их не видит — сессия smart-агента запускается без `--watch-db`. Здесь флаг есть.

Исключения запуска и обмена (FileNotFoundError, TimeoutError, ошибки протокола) НЕ
перехватываются — их переводит в сообщение на русском вызывающий код, как в
mcp_integration/tools_command.py. Ошибка ИНСТРУМЕНТА (сервер вернул isError) — это
WatchToolError с текстом сервера: он уже написан для чтения (что случилось и что
делать) и показывается пользователю как есть.

Чистые функции (сборка параметров запуска, разбор результата, разворачивание группы
исключений) покрыты tests/test_watch_session.py без запуска процессов.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCP_MOEX_DIR, PRICE_WATCH_DB, PRICE_WATCH_TIMEOUT_SECONDS

# MCP SDK заворачивает сообщение об ошибке инструмента в английский префикс
# («Error executing tool watch_set: <русский текст сервера>»). Пользователю показывается
# только текст сервера — в живой проверке префикс приходил в каждом ответе isError.
_SDK_ERROR_PREFIX = re.compile(r"^Error executing tool \w+:\s*")


class WatchToolError(Exception):
    """Сервер выполнил вызов и сообщил об ошибке (isError). `str(error)` — текст
    сервера, написанный для чтения: неизвестный тикер, границы периода и т.п."""


class WatchProtocolError(Exception):
    """Сервер ответил не так, как обещает контракт (пустой или не разбираемый результат)."""


def build_watch_server_params(directory: str, db_path: str) -> StdioServerParameters:
    """Параметры запуска сервера в режиме расписания.

    Команда собирается списком аргументов, без shell: значения directory и db_path не
    могут внедрить команду. db_path должен быть АБСОЛЮТНЫМ: `uv run --directory`
    меняет рабочий каталог процесса, и относительный путь указал бы внутрь каталога
    mcp-moex (config.PRICE_WATCH_DB делает путь абсолютным при загрузке).
    """
    return StdioServerParameters(
        command="uv",
        args=["run", "--directory", directory, "mcp-moex", "--watch-db", db_path],
    )


def result_to_dict(result: Any) -> dict:
    """Результат вызова инструмента → словарь `structured_content`.

    is_error — WatchToolError с текстом сервера. Если структурного результата нет,
    берётся текст блоков как JSON (клиенты без поддержки structuredContent получают
    то же самое дублем в тексте). Всё остальное — WatchProtocolError.
    """
    text_blocks = "".join(getattr(block, "text", "") or "" for block in (result.content or []))
    if getattr(result, "is_error", False):
        text = _SDK_ERROR_PREFIX.sub("", text_blocks).strip()
        raise WatchToolError(text or "Сервер сообщил об ошибке без пояснения.")

    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    if text_blocks:
        try:
            parsed = json.loads(text_blocks)
        except json.JSONDecodeError as exc:
            raise WatchProtocolError("Результат сервера не является JSON.") from exc
        if isinstance(parsed, dict):
            return parsed
    raise WatchProtocolError("Сервер вернул пустой или неожиданный результат.")


def unwrap_exception_group(error: BaseException) -> BaseException:
    """Группа исключений из ОДНОГО исключения → само исключение.

    Контексты stdio_client/ClientSession построены на anyio, который может завернуть
    исключение из тела в ExceptionGroup и обойти `except TimeoutError` и т.п. у
    вызывающего кода (пользователь увидел бы «непредвиденную ошибку» вместо понятного
    текста). Та же причина, что в SmartAgent._tool_completion_async. Проверка по
    атрибуту `exceptions`, а не по BaseExceptionGroup: тот появился только в Python
    3.11, а проект заявляет 3.10+ (pyproject.toml).
    """
    while True:
        nested = getattr(error, "exceptions", None)
        if not isinstance(nested, tuple | list) or len(nested) != 1:
            return error
        error = nested[0]


async def call_watch_tool(
    name: str,
    arguments: dict | None = None,
    *,
    directory: str = MCP_MOEX_DIR,
    db_path: str = PRICE_WATCH_DB,
    timeout: float = PRICE_WATCH_TIMEOUT_SECONDS,
) -> dict:
    """Вызывает один инструмент расписания и возвращает его результат словарём.

    Процесс сервера запускается на это обращение и завершается до возврата. Тайм-аут
    охватывает весь цикл (запуск, рукопожатие, вызов) — самое вероятное место
    зависания это старт `uv`. Выход из `async with` при тайм-ауте или отмене
    гарантированно завершает процесс сервера.
    """
    params = build_watch_server_params(directory, db_path)
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
        # call_tool у SDK 2.x может вернуть и не CallToolResult (например, запрос
        # дополнительного ввода) — для инструментов расписания это не ожидается.
        raise WatchProtocolError(f"Сервер вернул неожиданный ответ на вызов {name}.")
    return result_to_dict(result)
