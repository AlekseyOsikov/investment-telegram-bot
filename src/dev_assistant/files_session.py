"""Файловый слой ассистента разработчика: запуск MCP-сервера файлов проекта и безопасный фасад
для модели (design.md изменения add-dev-project-files, решения 1, 5, 6).

Что гарантирует модуль (инварианты, не убирать):
- модели отдаются ТОЛЬКО `list_dir`, `read_file`, `search_text`, `propose_change`
  (`FILES_TOOL_ALLOWLIST`); `apply_change` и `discard_change` нет ни в схеме, ни в обработке —
  запись выполняет код бота по кнопке (`run_button_action`, отдельный allowlist);
- корень проекта и каталог предложений модель не выбирает: серверу они приходят аргументами
  запуска, а из аргументов вызова отбрасывается всё, чего нет в схеме инструмента;
- `propose_change` не read-only (пишет предложение в staging), поэтому сессия открывается с
  `require_read_only=False`, а безопасность держит явный allowlist;
- модели возвращается краткий результат `propose_change` (id, путь, размер), полный diff показывает
  бот; запись предложения в реестр делает фасад;
- запуск — списком аргументов без shell, тем же интерпретатором.

Механику сессии (рукопожатие, тайм-ауты, вызовы) переиспользует `mcp_integration.market_session`.
Чистые функции и фасад на фейковой сессии — tests/test_dev_assistant_files.py."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import StdioServerParameters

from mcp_integration.market_session import (
    ToolOutcome,
    open_market_tools,
    parse_arguments,
    truncate_result,
)

from . import proposals

FILES_TOOL_ALLOWLIST = frozenset({"list_dir", "read_file", "search_text", "propose_change"})
BUTTON_TOOL_ALLOWLIST = frozenset({"apply_change", "discard_change"})
SERVER_SCRIPT = str(
    Path(__file__).resolve().parents[2] / "mcp_servers" / "project_files" / "server.py"
)
# Результат сервера читается целиком (diff нужен боту); усечение для модели — в фасаде.
INNER_MAX_RESULT_CHARS = 1_000_000


@dataclass(frozen=True)
class FilesSettings:
    """Настройки оператора для запуска сервера (значения берёт `command.py` из `config`)."""

    root: str
    staging_dir: str
    ttl_seconds: int
    max_read_chars: int
    max_search_results: int
    max_proposal_chars: int
    max_proposals: int = 3


def build_files_server_params(settings: FilesSettings) -> StdioServerParameters:
    """Запуск сервера тем же интерпретатором, списком аргументов без shell: значения не могут
    внедрить команду. Корень и каталог предложений — только из настроек оператора."""
    return StdioServerParameters(
        command=sys.executable,
        args=[
            SERVER_SCRIPT,
            "--root", settings.root,
            "--staging-dir", settings.staging_dir,
            "--ttl-seconds", str(settings.ttl_seconds),
            "--max-read-chars", str(settings.max_read_chars),
            "--max-search-results", str(settings.max_search_results),
            "--max-proposal-chars", str(settings.max_proposal_chars),
            "--max-proposals", str(settings.max_proposals),
        ],
    )


def schema_properties(tool: dict) -> set[str]:
    parameters = tool.get("function", {}).get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    return set(properties) if isinstance(properties, dict) else set()


def filter_arguments(arguments: dict, allowed: set[str]) -> dict:
    """Оставляет только параметры из схемы инструмента: присланные моделью `root`, `staging_dir`
    и прочее постороннее к серверу не доходят."""
    return {key: value for key, value in arguments.items() if key in allowed}


def model_proposal_result(proposal: proposals.Proposal) -> str:
    """Краткий результат `propose_change` для модели: без diff (его показывает бот)."""
    return json.dumps(
        {
            "proposal_id": proposal.id,
            "path": proposal.path,
            "creates": proposal.creates,
            "added": proposal.added,
            "removed": proposal.removed,
            "status": (
                "Предложение показано пользователю как diff с кнопками «Применить» и "
                "«Отклонить». Файл НЕ изменён; не утверждай, что он изменён."
            ),
        },
        ensure_ascii=False,
    )


class FilesTools:
    """Фасад над открытой сессией файлового сервера для модели. Совместим с `run_tool_loop`
    (`openai_tools`, `async call(name, raw) -> ToolOutcome`)."""

    def __init__(
        self,
        inner: Any,
        *,
        max_result_chars: int,
        registry: proposals.ProposalRegistry,
        chat_id: int,
        user_id: int,
    ) -> None:
        self._inner = inner
        self._max_result_chars = max_result_chars
        self._registry = registry
        self._chat_id = chat_id
        self._user_id = user_id
        self.openai_tools: list[dict] = [
            tool
            for tool in inner.openai_tools
            if tool.get("function", {}).get("name") in FILES_TOOL_ALLOWLIST
        ]
        self._names = [tool["function"]["name"] for tool in self.openai_tools]
        self._allowed_args = {
            tool["function"]["name"]: schema_properties(tool) for tool in self.openai_tools
        }
        self.proposals: list[proposals.Proposal] = []

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
        arguments, error = parse_arguments(raw_arguments)
        if error is not None or arguments is None:
            return ToolOutcome(error or "Аргументы вызова не разобраны.", is_error=True)
        arguments = filter_arguments(arguments, self._allowed_args[name])
        outcome = await self._inner.call(name, json.dumps(arguments, ensure_ascii=False))
        if name == "propose_change" and not outcome.is_error:
            return self._register(outcome)
        return ToolOutcome(
            truncate_result(outcome.text, self._max_result_chars),
            is_error=outcome.is_error,
            transport_failure=outcome.transport_failure,
        )

    def _register(self, outcome: ToolOutcome) -> ToolOutcome:
        try:
            data = proposals.parse_server_proposal(json.loads(outcome.text))
        except ValueError:
            data = None
        if data is None:
            return ToolOutcome(
                "Сервер вернул непонятный ответ на подготовку предложения: правка не показана "
                "пользователю. Опиши её текстом.",
                is_error=True,
            )
        proposal = self._registry.register(self._chat_id, self._user_id, data)
        self.proposals.append(proposal)
        return ToolOutcome(model_proposal_result(proposal))


@asynccontextmanager
async def open_files_tools(
    settings: FilesSettings,
    timeout: float,
    max_result_chars: int,
    registry: proposals.ProposalRegistry,
    chat_id: int,
    user_id: int,
) -> AsyncIterator[FilesTools]:
    """Запускает файловый сервер и отдаёт `FilesTools`; процесс завершается при выходе из блока.
    Исключения запуска не перехватываются (делает вызывающий код: слой тогда недоступен)."""
    async with open_market_tools(
        build_files_server_params(settings),
        timeout,
        INNER_MAX_RESULT_CHARS,
        FILES_TOOL_ALLOWLIST,
        False,
    ) as inner:
        yield FilesTools(
            inner,
            max_result_chars=max_result_chars,
            registry=registry,
            chat_id=chat_id,
            user_id=user_id,
        )


async def run_button_action(
    tool_name: str,
    proposal_id: str,
    settings: FilesSettings,
    timeout: float,
    open_tools: Callable[..., Any] = open_market_tools,
) -> ToolOutcome:
    """Применение или отклонение по кнопке: отдельная короткая сессия с allowlist
    `{apply_change, discard_change}`. Недоступна фасаду модели. Исключения запуска не
    перехватываются (делает вызывающий код)."""
    if tool_name not in BUTTON_TOOL_ALLOWLIST:
        raise ValueError(f"инструмент {tool_name!r} не предназначен для кнопок")
    if not proposals.is_valid_id(proposal_id):
        raise ValueError("недопустимый идентификатор предложения")
    async with open_tools(
        build_files_server_params(settings),
        timeout,
        INNER_MAX_RESULT_CHARS,
        BUTTON_TOOL_ALLOWLIST,
        False,
    ) as inner:
        return await inner.call(tool_name, json.dumps({"proposal_id": proposal_id}))


class CombinedTools:
    """Объединяет инструменты нескольких слоёв (git, файлы) в один набор для `run_tool_loop`.
    Имена слоёв не пересекаются; пересечение — ошибка конфигурации."""

    def __init__(self, *layers: Any) -> None:
        self._owner: dict[str, Any] = {}
        self.openai_tools: list[dict] = []
        for layer in layers:
            for tool in layer.openai_tools:
                name = tool["function"]["name"]
                if name in self._owner:
                    raise ValueError(f"инструмент {name!r} предоставлен двумя слоями")
                self._owner[name] = layer
                self.openai_tools.append(tool)

    async def call(self, name: str, raw_arguments: str | None) -> ToolOutcome:
        layer = self._owner.get(name)
        if layer is None:
            available = ", ".join(self._owner) or "нет"
            return ToolOutcome(
                f"Инструмента «{name}» нет среди доступных. Доступные инструменты: {available}.",
                is_error=True,
            )
        return await layer.call(name, raw_arguments)
