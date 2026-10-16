"""Тесты CRM-слоя поддержки (support_assistant/crm_session.py): схемы без telegram_user_id,
подстановка идентификатора, перечень инструментов, параметры запуска, предзагрузка и лимиты.
Без процессов: сессия подменена фейком."""

import asyncio
import json
import sys

from mcp_integration.market_session import ToolOutcome
from support_assistant import crm_session, rules
from support_assistant.crm_session import SUPPORT_TOOL_ALLOWLIST, CrmTools

USER = 777


def tool(name, properties=None, required=None):
    props = {"telegram_user_id": {"type": "integer"}, **(properties or {})}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": ["telegram_user_id", *(required or [])],
            },
        },
    }


class FakeInner:
    def __init__(self, names=("get_my_profile", "list_my_tickets", "get_ticket", "close_ticket")):
        self.openai_tools = [tool(n, {"ticket_id": {"type": "string"}}) for n in names]
        self.calls: list[tuple[str, str | None]] = []
        self.reply = ToolOutcome(json.dumps({"tickets": []}))

    async def call(self, name, raw):
        self.calls.append((name, raw))
        return self.reply


def run(coro):
    return asyncio.run(coro)


def ticket(ticket_id, status="open", updated="2026-10-10", **extra):
    base = {
        "ticket_id": ticket_id,
        "subject": "Тема",
        "bot_feature": "/watch",
        "status": status,
        "created": "2026-10-01",
        "updated": updated,
        "error_text": "",
        "events": [],
    }
    base.update(extra)
    return base


# --- перечень и схемы ------------------------------------------------------------------------


def test_allowlist_is_exactly_three_read_tools():
    assert SUPPORT_TOOL_ALLOWLIST == {"get_my_profile", "list_my_tickets", "get_ticket"}


def test_strip_user_param_without_mutating_source():
    source = tool("get_ticket", {"ticket_id": {"type": "string"}}, ["ticket_id"])
    stripped = crm_session.strip_user_param(source)
    params = stripped["function"]["parameters"]
    assert "telegram_user_id" not in params["properties"]
    assert params["required"] == ["ticket_id"]
    assert "telegram_user_id" in source["function"]["parameters"]["properties"]


def test_strip_user_param_drops_empty_required():
    stripped = crm_session.strip_user_param(tool("get_my_profile"))
    assert "required" not in stripped["function"]["parameters"]


def test_facade_exposes_only_allowlisted_tools_without_user_id():
    facade = CrmTools(FakeInner(), USER)
    assert facade.tool_names == ["get_my_profile", "list_my_tickets", "get_ticket"]
    for item in facade.openai_tools:
        assert "telegram_user_id" not in item["function"]["parameters"]["properties"]


# --- подстановка идентификатора --------------------------------------------------------------


def test_call_overwrites_foreign_user_id():
    inner = FakeInner()
    facade = CrmTools(inner, USER)
    args = json.dumps({"telegram_user_id": 1, "ticket_id": "T-2001"})
    outcome = run(facade.call("get_ticket", args))
    assert not outcome.is_error
    name, raw = inner.calls[0]
    assert name == "get_ticket"
    assert json.loads(raw) == {"telegram_user_id": USER, "ticket_id": "T-2001"}


def test_call_without_arguments_gets_user_id():
    inner = FakeInner()
    run(CrmTools(inner, USER).call("list_my_tickets", None))
    assert json.loads(inner.calls[0][1]) == {"telegram_user_id": USER}


def test_call_with_bad_json_is_error_and_not_forwarded():
    inner = FakeInner()
    outcome = run(CrmTools(inner, USER).call("get_ticket", "{не json"))
    assert outcome.is_error and inner.calls == []


def test_call_outside_allowlist_is_error_and_not_forwarded():
    inner = FakeInner()
    outcome = run(CrmTools(inner, USER).call("close_ticket", "{}"))
    assert outcome.is_error and "close_ticket" in outcome.text
    assert inner.calls == []


def test_display_arguments_hide_user_id():
    shown = crm_session.display_arguments(json.dumps({"telegram_user_id": 5, "ticket_id": "T-1"}))
    assert "telegram_user_id" not in shown and "T-1" in shown
    assert crm_session.display_arguments(json.dumps({"telegram_user_id": 5})) == ""


# --- параметры запуска -----------------------------------------------------------------------


def test_mock_params_use_current_interpreter_and_demo_id():
    params = crm_session.build_crm_server_params("", [], True, 123, mock_script="/x/server.py")
    assert params.command == sys.executable
    assert params.args == ["/x/server.py", "--demo-user-id", "123"]


def test_mock_without_demo_id_has_no_flag():
    params = crm_session.build_crm_server_params("", [], True, None, mock_script="/x/server.py")
    assert params.args == ["/x/server.py"]


def test_external_command_has_priority_and_no_demo_flag():
    params = crm_session.build_crm_server_params(
        "uvx", ["crm-server", "--x"], True, 123, mock_script="/x/server.py"
    )
    assert params.command == "uvx" and params.args == ["crm-server", "--x"]


def test_nothing_configured_is_none():
    assert crm_session.build_crm_server_params("", [], False, 123) is None


# --- предзагрузка и лимиты -------------------------------------------------------------------


def test_select_open_tickets_skips_closed_sorts_and_limits():
    tickets = [
        ticket("T-1", updated="2026-10-01"),
        ticket("T-2", status="closed", updated="2026-10-20"),
        ticket("T-3", status="pending", updated="2026-10-15"),
        ticket("T-4", updated="2026-10-12"),
    ]
    selected, omitted = crm_session.select_open_tickets(tickets, 2)
    assert [t["ticket_id"] for t in selected] == ["T-3", "T-4"]
    assert omitted == 1


def test_select_open_tickets_limit_zero_is_empty():
    selected, omitted = crm_session.select_open_tickets([ticket("T-1")], 0)
    assert selected == [] and omitted == 1


def test_clamp_text_truncates_with_mark_within_limit():
    long = "а" * 1000
    clamped = crm_session.clamp_text(long)
    assert len(clamped) <= rules.TICKET_FIELD_MAX_CHARS
    assert clamped.endswith(rules.TRUNCATION_MARK)
    assert crm_session.clamp_text("коротко") == "коротко"
    assert crm_session.clamp_text(None) == ""


def test_normalize_ticket_truncates_fields_and_keeps_latest_events():
    events = [{"date": f"2026-10-{i:02d}", "text": "е" * 1000} for i in range(1, 10)]
    normalized = crm_session.normalize_ticket(
        ticket("T-1", error_text="о" * 1000, events=events, secret="не должно попасть")
    )
    assert "secret" not in normalized
    assert len(normalized["error_text"]) <= rules.TICKET_FIELD_MAX_CHARS
    assert len(normalized["events"]) == rules.TICKET_EVENTS_MAX
    assert normalized["events"][-1]["date"] == "2026-10-09"
    assert all(len(e["text"]) <= rules.TICKET_FIELD_MAX_CHARS for e in normalized["events"])


def test_parse_tickets_rejects_garbage():
    assert crm_session.parse_tickets("не json") is None
    assert crm_session.parse_tickets(json.dumps([1, 2])) is None
    assert crm_session.parse_tickets(json.dumps({"tickets": "x"})) is None
    assert crm_session.parse_tickets(json.dumps({"tickets": [ticket("T-1"), 5]})) == [
        ticket("T-1")
    ]


def test_preload_ok_uses_facade_and_user_id():
    inner = FakeInner()
    inner.reply = ToolOutcome(
        json.dumps({"tickets": [ticket("T-1"), ticket("T-2", status="closed")]})
    )
    snapshot = run(crm_session.preload_tickets(CrmTools(inner, USER), 5))
    assert snapshot.status == crm_session.TICKETS_OK
    assert [t["ticket_id"] for t in snapshot.tickets] == ["T-1"]
    assert json.loads(inner.calls[0][1]) == {"telegram_user_id": USER}


def test_preload_empty_list_is_ok_with_no_tickets():
    snapshot = run(crm_session.preload_tickets(CrmTools(FakeInner(), USER), 5))
    assert snapshot.status == crm_session.TICKETS_OK and snapshot.tickets == ()


def test_preload_tool_error_is_unavailable():
    inner = FakeInner()
    inner.reply = ToolOutcome("сбой", is_error=True, transport_failure=True)
    snapshot = run(crm_session.preload_tickets(CrmTools(inner, USER), 5))
    assert snapshot.status == crm_session.TICKETS_UNAVAILABLE


def test_preload_unparseable_is_unavailable():
    inner = FakeInner()
    inner.reply = ToolOutcome("обрезано {\"tickets\": [")
    snapshot = run(crm_session.preload_tickets(CrmTools(inner, USER), 5))
    assert snapshot.status == crm_session.TICKETS_UNAVAILABLE


def test_preload_without_list_tool_is_unavailable():
    facade = CrmTools(FakeInner(names=("get_my_profile",)), USER)
    snapshot = run(crm_session.preload_tickets(facade, 5))
    assert snapshot.status == crm_session.TICKETS_UNAVAILABLE
