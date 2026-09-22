"""Тесты форматирования результата MCP (mcp_integration/client.py: format_tools_result).

Как и agents/task_state.py, форматирование — чистая функция без сети и без
Telegram: здесь нет ни моков MCP-сессии, ни запущенных дочерних процессов, только
готовые dataclass-объекты McpToolsResult/McpTool на входе (тот же принцип, что
apply_state_response в тестах test_task_state.py — тестируется реальная функция
на собранных вручную данных, а не подставные заглушки).
"""

from mcp_integration.client import McpTool, McpToolsResult, format_tools_result


def test_format_nonempty_list():
    result = McpToolsResult(
        server_name="mcp-servers/everything",
        server_version="2.0.0",
        tools=[
            McpTool(name="echo", description="Echoes back the input string", required_params=["message"]),
            McpTool(name="get-env", description="Returns all environment variables", required_params=[]),
        ],
    )
    text = format_tools_result(result)

    assert "mcp-servers/everything" in text
    assert "2.0.0" in text
    assert "Доступно инструментов: 2" in text
    assert "echo" in text
    assert "Echoes back the input string" in text
    assert "Обязательные параметры: message" in text
    assert "get-env" in text


def test_format_empty_list():
    result = McpToolsResult(server_name="empty-server", server_version="1.0.0", tools=[])
    text = format_tools_result(result)

    assert "empty-server" in text
    assert "не предоставляет ни одного инструмента" in text


def test_format_tool_without_description():
    result = McpToolsResult(
        server_name="srv",
        server_version=None,
        tools=[McpTool(name="mystery-tool", description=None, required_params=[])],
    )
    text = format_tools_result(result)

    assert "mystery-tool" in text
    # Без описания и без версии сервера — соответствующие строки просто не появляются.
    assert "srv" in text


def test_format_tool_with_required_params():
    result = McpToolsResult(
        server_name="srv",
        server_version="1.0.0",
        tools=[
            McpTool(
                name="build-portfolio",
                description="Собирает портфель",
                required_params=["goal", "horizon", "risk"],
            )
        ],
    )
    text = format_tools_result(result)

    assert "Обязательные параметры: goal, horizon, risk" in text


def test_format_truncates_long_description():
    long_description = "x" * 500
    result = McpToolsResult(
        server_name="srv",
        server_version="1.0.0",
        tools=[McpTool(name="verbose-tool", description=long_description, required_params=[])],
    )
    text = format_tools_result(result)

    assert long_description not in text
    assert "…" in text
    # Строка описания короче исходной как минимум на 300 символов (усечена).
    description_line = next(line for line in text.splitlines() if line.strip().startswith("x"))
    assert len(description_line.strip()) < len(long_description)


def test_format_server_without_version():
    result = McpToolsResult(
        server_name="srv-no-version",
        server_version=None,
        tools=[McpTool(name="tool", description=None, required_params=[])],
    )
    text = format_tools_result(result)

    header_line = text.splitlines()[0]
    assert header_line == "🔌 Сервер: srv-no-version"
