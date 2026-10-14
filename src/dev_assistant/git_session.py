"""Git-слой ассистента разработчика: запуск read-only MCP-сервера `mcp-git` и безопасный фасад
для модели (design.md изменения add-dev-assistant-mode, решение 5).

Что гарантирует модуль (инварианты, не убирать):
- модели отдаются ТОЛЬКО `get_current_branch` и `list_files` (`GIT_TOOL_ALLOWLIST`); `get_diff` и
  `list_roots` нет ни в схеме, ни в обработке вызова — содержимое изменений в модель не попадает;
- `repo_path` модель не выбирает: параметр удалён из схем (`strip_repo_path`), а в каждом вызове
  перезаписывается значением `root` (`force_repo_path`), даже если модель его прислала;
- корень доступа сервера (`--root`) — каталог этого репозитория, поэтому и подмена `repo_path`
  упёрлась бы в `path_not_allowed` на стороне сервера;
- запуск — списком аргументов без shell; каталог проекта сервера задаёт оператор (`MCP_GIT_DIR`).

Механику сессии (рукопожатие, тайм-ауты, усечение, read_only_hint) переиспользует
`mcp_integration.market_session`. Чистые функции — tests/test_dev_assistant_git.py."""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp import StdioServerParameters

from mcp_integration.market_session import (
    ToolOutcome,
    open_market_tools,
    parse_arguments,
)

GIT_TOOL_ALLOWLIST = frozenset({"get_current_branch", "list_files"})
REPO_PATH_PARAM = "repo_path"


def build_git_server_params(directory: str, root: str) -> StdioServerParameters:
    """`uv run --directory <directory> mcp-git --root <root>` — списком аргументов без shell,
    значения не могут внедрить команду. Корень доступа сервера — только `root`."""
    return StdioServerParameters(
        command="uv", args=["run", "--directory", directory, "mcp-git", "--root", root]
    )


def strip_repo_path(tool: dict) -> dict:
    """Копия описания инструмента (формат `tools` OpenAI) без параметра `repo_path`:
    ни в `properties`, ни в `required`. Модель не видит параметра и не может его выбрать."""
    result = copy.deepcopy(tool)
    parameters = result.get("function", {}).get("parameters")
    if isinstance(parameters, dict):
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            properties.pop(REPO_PATH_PARAM, None)
        required = parameters.get("required")
        if isinstance(required, list):
            parameters["required"] = [name for name in required if name != REPO_PATH_PARAM]
            if not parameters["required"]:
                del parameters["required"]
    return result


def force_repo_path(raw_arguments: str | None, root: str) -> tuple[dict | None, str | None]:
    """Разбирает аргументы вызова и подставляет `repo_path=root`, перезаписывая присланное
    моделью значение. Возвращает (аргументы, None) либо (None, текст ошибки для модели)."""
    arguments, error = parse_arguments(raw_arguments)
    if error is not None or arguments is None:
        return None, error
    arguments = dict(arguments)
    arguments[REPO_PATH_PARAM] = root
    return arguments, None


def display_arguments(raw_arguments: str | None) -> str:
    """Аргументы вызова для показа пользователю: без `repo_path` (его выбирает не модель, и
    присланное ею значение к вызову не применяется)."""
    arguments, error = parse_arguments(raw_arguments)
    if error is not None or arguments is None:
        return raw_arguments or ""
    arguments = {k: v for k, v in arguments.items() if k != REPO_PATH_PARAM}
    return json.dumps(arguments, ensure_ascii=False) if arguments else ""


class GitTools:
    """Фасад над открытой сессией `mcp-git` для модели: схемы без `repo_path`, вызовы только
    разрешённых инструментов с подставленным `repo_path`. Совместим с `run_tool_loop`
    (`openai_tools`, `async call(name, raw) -> ToolOutcome`)."""

    def __init__(self, inner: Any, root: str) -> None:
        self._inner = inner
        self._root = root
        self.openai_tools: list[dict] = [
            strip_repo_path(tool)
            for tool in inner.openai_tools
            if tool.get("function", {}).get("name") in GIT_TOOL_ALLOWLIST
        ]
        self._names = [tool["function"]["name"] for tool in self.openai_tools]

    @property
    def tool_names(self) -> list[str]:
        return list(self._names)

    async def call(self, name: str, raw_arguments: str | None) -> ToolOutcome:
        if name not in self._names:
            available = ", ".join(self._names) or "нет"
            return ToolOutcome(
                f"Инструмента «{name}» нет среди доступных. Доступные инструменты: {available}.",
                is_error=True,
            )
        arguments, error = force_repo_path(raw_arguments, self._root)
        if error is not None or arguments is None:
            return ToolOutcome(error or "Аргументы вызова не разобраны.", is_error=True)
        return await self._inner.call(name, json.dumps(arguments, ensure_ascii=False))


@asynccontextmanager
async def open_git_tools(
    directory: str, root: str, timeout: float, max_result_chars: int
) -> AsyncIterator[GitTools]:
    """Запускает `mcp-git` и отдаёт `GitTools`; процесс завершается при выходе из блока.
    Исключения запуска не перехватываются (делает вызывающий код: git-слой тогда считается
    недоступным). Пустой `openai_tools` — сервер не отдал ни одного разрешённого read-only
    инструмента; вызывающий код трактует это как недоступность."""
    async with open_market_tools(
        build_git_server_params(directory, root),
        timeout,
        max_result_chars,
        GIT_TOOL_ALLOWLIST,
        True,
    ) as inner:
        yield GitTools(inner, root)
