"""CRM-слой ассистента поддержки: запуск MCP-сервера «CRM» (мок или внешний), безопасный фасад для
модели и предзагрузка обращений пользователя (design.md изменения add-support-assistant, решения
2–5).

Что гарантирует модуль (инварианты, не убирать):
- модели отдаются ТОЛЬКО `get_my_profile`, `list_my_tickets`, `get_ticket`
  (`SUPPORT_TOOL_ALLOWLIST`) и только с пометкой read-only; любой иной вызов возвращается
  модели ошибкой;
- идентификатор пользователя модель не выбирает: параметр `telegram_user_id` удалён из схем
  (`strip_user_param`), а в каждом вызове перезаписывается значением из Telegram
  (`force_user_param`), даже если модель его прислала; это id ПОЛЬЗОВАТЕЛЯ, не чата;
- предзагрузка списка обращений идёт через тот же фасад, поэтому тоже только для этого
  пользователя; число обращений и длина полей ограничены;
- запуск — списком аргументов без shell; способ запуска задаёт оператор.

Механику сессии (рукопожатие, тайм-ауты, усечение, read_only_hint) переиспользует
`mcp_integration.market_session`. Чистые функции — tests/test_support_assistant_crm.py."""

from __future__ import annotations

import copy
import json
import os
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp import StdioServerParameters

from mcp_integration.market_session import (
    ToolOutcome,
    open_market_tools,
    parse_arguments,
)

from . import rules
from .sources import PROJECT_ROOT

SUPPORT_TOOL_ALLOWLIST = frozenset({"get_my_profile", "list_my_tickets", "get_ticket"})
USER_PARAM = "telegram_user_id"
LIST_TOOL = "list_my_tickets"
CLOSED_STATUS = "closed"

MOCK_SERVER_SCRIPT = os.path.join(PROJECT_ROOT, "mock_servers", "support_crm", "server.py")

# Статусы предзагрузки обращений.
TICKETS_OK = "ok"
TICKETS_UNAVAILABLE = "unavailable"  # CRM настроена, но не ответила / сбой
TICKETS_NOT_CONFIGURED = "not_configured"


def build_crm_server_params(
    command: str,
    args: Sequence[str],
    use_mock: bool,
    demo_user_id: int | None,
    mock_script: str = MOCK_SERVER_SCRIPT,
) -> StdioServerParameters | None:
    """Параметры запуска CRM-сервера или None (CRM не настроена). Внешняя CRM
    (`MCP_SUPPORT_COMMAND`/`MCP_SUPPORT_ARGS`) приоритетнее мока; `--demo-user-id` добавляется
    ТОЛЬКО к встроенному моку. Мок запускается тем же интерпретатором, что и бот. Всё — списком
    аргументов без shell."""
    if command:
        return StdioServerParameters(command=command, args=list(args))
    if not use_mock:
        return None
    mock_args = [mock_script]
    if demo_user_id is not None:
        mock_args += ["--demo-user-id", str(demo_user_id)]
    return StdioServerParameters(command=sys.executable, args=mock_args)


def strip_user_param(tool: dict) -> dict:
    """Копия описания инструмента (формат `tools` OpenAI) без параметра `telegram_user_id`:
    ни в `properties`, ни в `required`. Модель не видит параметра и не может его выбрать."""
    result = copy.deepcopy(tool)
    parameters = result.get("function", {}).get("parameters")
    if isinstance(parameters, dict):
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            properties.pop(USER_PARAM, None)
        required = parameters.get("required")
        if isinstance(required, list):
            parameters["required"] = [name for name in required if name != USER_PARAM]
            if not parameters["required"]:
                del parameters["required"]
    return result


def force_user_param(raw_arguments: str | None, user_id: int) -> tuple[dict | None, str | None]:
    """Разбирает аргументы вызова и подставляет `telegram_user_id=user_id`, перезаписывая
    присланное моделью значение. Возвращает (аргументы, None) либо (None, текст ошибки)."""
    arguments, error = parse_arguments(raw_arguments)
    if error is not None or arguments is None:
        return None, error
    arguments = dict(arguments)
    arguments[USER_PARAM] = user_id
    return arguments, None


def display_arguments(raw_arguments: str | None) -> str:
    """Аргументы вызова для показа пользователю: без `telegram_user_id`."""
    arguments, error = parse_arguments(raw_arguments)
    if error is not None or arguments is None:
        return raw_arguments or ""
    arguments = {k: v for k, v in arguments.items() if k != USER_PARAM}
    return json.dumps(arguments, ensure_ascii=False) if arguments else ""


class CrmTools:
    """Фасад над открытой сессией CRM для модели ОДНОГО пользователя: схемы без
    `telegram_user_id`, вызовы только разрешённых инструментов с подставленным идентификатором.
    Совместим с `run_tool_loop` (`openai_tools`, `async call(name, raw) -> ToolOutcome`)."""

    def __init__(self, inner: Any, user_id: int) -> None:
        self._inner = inner
        self._user_id = user_id
        self.openai_tools: list[dict] = [
            strip_user_param(tool)
            for tool in inner.openai_tools
            if tool.get("function", {}).get("name") in SUPPORT_TOOL_ALLOWLIST
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
        arguments, error = force_user_param(raw_arguments, self._user_id)
        if error is not None or arguments is None:
            return ToolOutcome(error or "Аргументы вызова не разобраны.", is_error=True)
        return await self._inner.call(name, json.dumps(arguments, ensure_ascii=False))


@asynccontextmanager
async def open_crm_tools(
    params: StdioServerParameters, user_id: int, timeout: float, max_result_chars: int
) -> AsyncIterator[CrmTools]:
    """Запускает CRM-сервер и отдаёт `CrmTools` для пользователя `user_id`; процесс завершается
    при выходе из блока. Исключения запуска не перехватываются (делает вызывающий код: CRM тогда
    считается недоступной). Пустой `openai_tools` — сервер не отдал ни одного разрешённого
    read-only инструмента; вызывающий код трактует это как недоступность."""
    async with open_market_tools(
        params, timeout, max_result_chars, SUPPORT_TOOL_ALLOWLIST, True
    ) as inner:
        yield CrmTools(inner, user_id)


# --- предзагрузка обращений ---------------------------------------------------------------


@dataclass(frozen=True)
class TicketsSnapshot:
    """Итог предзагрузки: `status` — `TICKETS_*`; `tickets` — самые свежие открытые обращения
    (уже нормализованные); `omitted` — сколько открытых не вошло из-за лимита."""

    status: str
    tickets: tuple[dict, ...] = ()
    omitted: int = 0
    failure: str | None = None


def clamp_text(value: Any, max_chars: int = rules.TICKET_FIELD_MAX_CHARS) -> str:
    """Текстовое поле обращения → строка не длиннее `max_chars` с пометкой об усечении."""
    text = "" if value is None else str(value).strip()
    if len(text) <= max_chars:
        return text
    budget = max(max_chars - len(rules.TRUNCATION_MARK), 0)
    return text[:budget].rstrip() + rules.TRUNCATION_MARK


def normalize_ticket(ticket: dict) -> dict:
    """Обращение для подачи модели: только известные поля, строки усечены, событий не больше
    `TICKET_EVENTS_MAX` (самые свежие — в конце списка CRM)."""
    events_raw = ticket.get("events")
    events = []
    if isinstance(events_raw, list):
        for event in events_raw[-rules.TICKET_EVENTS_MAX :]:
            if isinstance(event, dict):
                events.append(
                    {
                        "date": clamp_text(event.get("date"), 20),
                        "text": clamp_text(event.get("text")),
                    }
                )
    return {
        "ticket_id": clamp_text(ticket.get("ticket_id"), 40),
        "subject": clamp_text(ticket.get("subject")),
        "bot_feature": clamp_text(ticket.get("bot_feature"), 80),
        "status": clamp_text(ticket.get("status"), 20),
        "created": clamp_text(ticket.get("created"), 20),
        "updated": clamp_text(ticket.get("updated"), 20),
        "error_text": clamp_text(ticket.get("error_text")),
        "events": events,
    }


def parse_tickets(text: str) -> list[dict] | None:
    """Ответ `list_my_tickets` → список обращений либо None (не разобран / не тот формат)."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("tickets"), list):
        return None
    return [item for item in payload["tickets"] if isinstance(item, dict)]


def select_open_tickets(tickets: Sequence[dict], limit: int) -> tuple[list[dict], int]:
    """Открытые (не закрытые) обращения, самые свежие первыми, не больше `limit`. Возвращает
    (отобранные нормализованные, сколько открытых не вошло)."""
    open_tickets = [t for t in tickets if str(t.get("status", "")).lower() != CLOSED_STATUS]
    open_tickets.sort(
        key=lambda t: (str(t.get("updated", "")), str(t.get("ticket_id", ""))), reverse=True
    )
    limit = max(limit, 0)
    return [normalize_ticket(t) for t in open_tickets[:limit]], max(len(open_tickets) - limit, 0)


async def preload_tickets(crm: Any, limit: int) -> TicketsSnapshot:
    """Один вызов `list_my_tickets` через фасад (идентификатор подставляет он) → снимок. CRM без
    нужного инструмента, ошибка инструмента и неразобранный ответ — `TICKETS_UNAVAILABLE`."""
    if LIST_TOOL not in crm.tool_names:
        return TicketsSnapshot(TICKETS_UNAVAILABLE, failure="нет инструмента списка обращений")
    outcome = await crm.call(LIST_TOOL, "{}")
    if outcome.is_error:
        return TicketsSnapshot(TICKETS_UNAVAILABLE, failure="ошибка инструмента списка обращений")
    tickets = parse_tickets(outcome.text)
    if tickets is None:
        return TicketsSnapshot(TICKETS_UNAVAILABLE, failure="ответ списка обращений не разобран")
    selected, omitted = select_open_tickets(tickets, limit)
    return TicketsSnapshot(TICKETS_OK, tuple(selected), omitted)

