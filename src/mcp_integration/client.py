"""Подключение к MCP-серверу по stdio и получение списка его инструментов.

Один вызов connect_and_list_tools() = один цикл «запустить дочерний процесс сервера
→ рукопожатие протокола → list_tools() → закрыть процесс». Сессия не кэшируется
между вызовами (в отличие от, например, _agents в agents/agent_command.py) —
диагностическая команда /mcp_tools вызывается редко, а удержание процесса живым
между вызовами потребовало бы отдельно следить за его здоровьем и перезапускать
упавший сервер. Если это когда-нибудь станет узким местом, пересмотр не требует
изменений в спеке — см. design.md изменения add-mcp-tools-listing.

Эта функция НЕ перехватывает исключения — как Agent.ask() в agents/agent.py,
она отдаёт их вызывающему коду (mcp_integration/tools_command.py) как есть:
FileNotFoundError (команда запуска не найдена), TimeoutError/asyncio.TimeoutError
(не уложились в MCP_TIMEOUT_SECONDS — тайм-аут оборачивает весь цикл, а не только
list_tools(), т.к. зависнуть может уже сам запуск npx при первом скачивании
пакета), а также любые исключения самого MCP SDK при нарушении протокола сервером.
Перевод в сообщение на русском — забота tools_command.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCP_SERVER_ARGS, MCP_SERVER_COMMAND, MCP_TIMEOUT_SECONDS


@dataclass(frozen=True)
class McpTool:
    """Один инструмент, о котором сообщил MCP-сервер."""

    name: str
    description: str | None
    required_params: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class McpToolsResult:
    """Результат успешного подключения и запроса списка инструментов."""

    server_name: str
    server_version: str | None
    tools: list[McpTool]


async def connect_and_list_tools() -> McpToolsResult:
    """Подключается к MCP-серверу, заданному конфигурацией (config.py:
    MCP_SERVER_COMMAND/MCP_SERVER_ARGS), и возвращает его инструменты.

    Пользователь чата не может повлиять на то, к какому серверу идёт подключение —
    команда/аргументы читаются только из конфигурации оператора (см. докстринг
    MCP_SERVER_COMMAND в config.py и «Ограничения безопасности» в CLAUDE.md).

    Весь цикл — запуск процесса, рукопожатие (ClientSession.initialize()) и
    list_tools() — обёрнут одним asyncio.timeout(MCP_TIMEOUT_SECONDS): выход из
    async with при отмене по тайм-ауту гарантированно завершает дочерний процесс,
    отдельного кода его принудительного убийства не требуется.
    """
    params = StdioServerParameters(command=MCP_SERVER_COMMAND, args=MCP_SERVER_ARGS)
    async with asyncio.timeout(MCP_TIMEOUT_SECONDS):
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                init_result = await session.initialize()
                list_result = await session.list_tools()

    tools = [
        McpTool(
            name=tool.name,
            description=tool.description,
            required_params=list((tool.input_schema or {}).get("required") or []),
        )
        for tool in list_result.tools
    ]
    return McpToolsResult(
        server_name=init_result.server_info.name,
        server_version=init_result.server_info.version,
        tools=tools,
    )


# Описание инструмента усекается до этой длины — сторонние MCP-серверы не
# ограничены протоколом в длине description, а сообщение Telegram должно
# оставаться читаемым даже при большом числе инструментов.
_DESCRIPTION_MAX_CHARS = 200


def format_tools_result(result: McpToolsResult) -> str:
    """Форматирует результат connect_and_list_tools() в текст для пользователя.

    Чистая функция без сети и без Telegram — по тому же принципу, что
    task_state.describe_state (agents/task_state.py) не знает ни про LLM, ни про
    Telegram: она только собирает текст из уже готовых данных. Это и позволяет
    покрыть форматирование юнит-тестами (tests/test_mcp_formatting.py) без мока
    MCP-сессии.
    """
    version_suffix = f" {result.server_version}" if result.server_version else ""
    header = f"🔌 Сервер: {result.server_name}{version_suffix}"

    if not result.tools:
        return (
            f"{header}\n\n"
            "Соединение установлено, но сервер не предоставляет ни одного инструмента."
        )

    lines = [header, f"🧰 Доступно инструментов: {len(result.tools)}", ""]
    for i, tool in enumerate(result.tools, start=1):
        lines.append(f"{i}. {tool.name}")
        if tool.description:
            description = tool.description.strip()
            if len(description) > _DESCRIPTION_MAX_CHARS:
                description = description[:_DESCRIPTION_MAX_CHARS].rstrip() + "…"
            lines.append(f"   {description}")
        if tool.required_params:
            lines.append(f"   Обязательные параметры: {', '.join(tool.required_params)}")

    return "\n".join(lines)
