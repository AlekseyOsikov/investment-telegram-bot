"""Тесты файлового слоя /dev (dev_assistant/files_session.py): параметры запуска, allowlist,
отсутствие apply/discard у модели, отброс чужих аргументов, регистрация предложений, кнопки.
Без процессов: сессия подменена фейком."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager

import pytest

from dev_assistant import files_session, proposals
from dev_assistant.files_session import (
    BUTTON_TOOL_ALLOWLIST,
    FILES_TOOL_ALLOWLIST,
    CombinedTools,
    FilesSettings,
    FilesTools,
)
from mcp_integration.market_session import ToolOutcome

PID = "0123456789ab"
SETTINGS = FilesSettings("/srv/project", "/srv/stage", 600, 1000, 20, 5000)


def tool(name, properties):
    return {
        "type": "function",
        "function": {
            "name": name, "description": name,
            "parameters": {"type": "object", "properties": properties},
        },
    }


S = {"type": "string"}
ALL_TOOLS = [
    tool("list_dir", {"path": S}),
    tool("read_file", {"path": S}),
    tool("search_text", {"query": S}),
    tool("propose_change", {"path": S, "new_content": S, "summary": S}),
    tool("apply_change", {"proposal_id": S}),
    tool("discard_change", {"proposal_id": S}),
]

PROPOSE_RESULT = json.dumps(
    {"proposal_id": PID, "path": "README.md", "creates": False, "added": 1, "removed": 1,
     "summary": "s", "base_hash": "h", "diff": "diff --git a/README.md b/README.md\n+x\n-y\n"}
)


class FakeInner:
    def __init__(self, propose_text=PROPOSE_RESULT, tools=None):
        self.openai_tools = list(ALL_TOOLS if tools is None else tools)
        self.calls: list[tuple[str, dict]] = []
        self.propose_text = propose_text

    async def call(self, name, raw):
        self.calls.append((name, json.loads(raw)))
        if name == "propose_change":
            return ToolOutcome(self.propose_text)
        return ToolOutcome("x" * 500)


def make_tools(inner=None, max_chars=100):
    registry = proposals.ProposalRegistry(600)
    inner = inner or FakeInner()
    return FilesTools(
        inner, max_result_chars=max_chars, registry=registry, chat_id=10, user_id=20
    ), inner, registry


def run(coro):
    return asyncio.run(coro)


def test_allowlists_are_exact_and_disjoint():
    assert FILES_TOOL_ALLOWLIST == {"list_dir", "read_file", "search_text", "propose_change"}
    assert BUTTON_TOOL_ALLOWLIST == {"apply_change", "discard_change"}
    assert not FILES_TOOL_ALLOWLIST & BUTTON_TOOL_ALLOWLIST


def test_server_params_no_shell_and_operator_values():
    params = files_session.build_files_server_params(SETTINGS)
    assert params.command == sys.executable
    assert params.args[0].endswith("mcp_servers/project_files/server.py")
    pairs = dict(zip(params.args[1::2], params.args[2::2], strict=False))
    assert pairs["--root"] == "/srv/project" and pairs["--staging-dir"] == "/srv/stage"
    assert pairs["--ttl-seconds"] == "600" and pairs["--max-proposals"] == "3"


def test_model_schema_has_no_apply_or_discard():
    facade, _, _ = make_tools()
    names = [t["function"]["name"] for t in facade.openai_tools]
    assert names == ["list_dir", "read_file", "search_text", "propose_change"]
    assert "apply_change" not in names and "discard_change" not in names


def test_model_cannot_call_apply_or_discard():
    facade, inner, _ = make_tools()
    for name in ("apply_change", "discard_change", "get_diff"):
        outcome = run(facade.call(name, json.dumps({"proposal_id": PID})))
        assert outcome.is_error and "нет среди доступных" in outcome.text
    assert inner.calls == []


def test_foreign_arguments_are_not_forwarded():
    facade, inner, _ = make_tools()
    arguments = {"path": "README.md", "root": "/etc", "staging_dir": "/x"}
    run(facade.call("read_file", json.dumps(arguments)))
    assert inner.calls == [("read_file", {"path": "README.md"})]


def test_bad_arguments_json_is_error_without_call():
    facade, inner, _ = make_tools()
    assert run(facade.call("read_file", "{не json")).is_error
    assert inner.calls == []


def test_long_results_truncated_for_model():
    facade, _, _ = make_tools(max_chars=100)
    outcome = run(facade.call("read_file", json.dumps({"path": "a"})))
    assert len(outcome.text) <= 100 and "ОБРЕЗАН" in outcome.text


def test_propose_registers_and_hides_diff_from_model():
    facade, _, registry = make_tools()
    outcome = run(
        facade.call("propose_change", json.dumps({"path": "README.md", "new_content": "x"}))
    )
    assert not outcome.is_error
    assert "diff" not in json.loads(outcome.text) and "+x" not in outcome.text
    assert PID in outcome.text and "НЕ изменён" in outcome.text
    [proposal] = facade.proposals
    assert proposal.diff.startswith("diff --git")
    assert proposal.chat_id == 10 and proposal.user_id == 20
    assert registry.get(PID) == proposal


def test_propose_error_is_passed_and_not_registered():
    class Failing(FakeInner):
        async def call(self, name, raw):
            return ToolOutcome("В этот путь запись недопустима", is_error=True)

    facade, _, registry = make_tools(Failing())
    arguments = {"path": "src/main.py", "new_content": "x"}
    outcome = run(facade.call("propose_change", json.dumps(arguments)))
    assert outcome.is_error and facade.proposals == [] and len(registry) == 0


@pytest.mark.parametrize("text", ["не json", "[]", json.dumps({"proposal_id": "bad"})])
def test_propose_garbled_server_answer_is_error(text):
    facade, _, registry = make_tools(FakeInner(propose_text=text))
    arguments = {"path": "README.md", "new_content": "x"}
    outcome = run(facade.call("propose_change", json.dumps(arguments)))
    assert outcome.is_error and facade.proposals == [] and len(registry) == 0


def test_inner_without_allowed_tools_gives_empty_facade():
    facade, _, _ = make_tools(FakeInner(tools=[ALL_TOOLS[4]]))
    assert facade.openai_tools == []


# --- кнопки ---------------------------------------------------------------------------------


def make_fake_opener(calls, answer=None):
    answer = answer or ToolOutcome("{}")
    class Inner:
        async def call(self, name, raw):
            calls.append((name, json.loads(raw)))
            return answer

    @asynccontextmanager
    async def opener(params, timeout, max_chars, allowlist, require_read_only):
        calls.append(("open", sorted(allowlist), require_read_only))
        yield Inner()

    return opener


def test_button_action_uses_separate_allowlist():
    calls = []
    outcome = run(
        files_session.run_button_action(
            "apply_change", PID, SETTINGS, 5, open_tools=make_fake_opener(calls)
        )
    )
    assert outcome.text == "{}"
    assert calls[0] == ("open", ["apply_change", "discard_change"], False)
    assert calls[1] == ("apply_change", {"proposal_id": PID})


@pytest.mark.parametrize("tool_name", ["propose_change", "read_file", "get_diff"])
def test_button_action_rejects_model_tools(tool_name):
    calls = []
    with pytest.raises(ValueError):
        run(
            files_session.run_button_action(
                tool_name, PID, SETTINGS, 5, open_tools=make_fake_opener(calls)
            )
        )
    assert calls == []


def test_button_action_rejects_bad_id():
    calls = []
    with pytest.raises(ValueError):
        run(
            files_session.run_button_action(
                "apply_change", "../x", SETTINGS, 5, open_tools=make_fake_opener(calls)
            )
        )
    assert calls == []


# --- объединение слоёв ---------------------------------------------------------------------


class Layer:
    def __init__(self, *names):
        self.openai_tools = [tool(n, {}) for n in names]
        self.calls = []

    async def call(self, name, raw):
        self.calls.append(name)
        return ToolOutcome(name)


def test_combined_tools_dispatch_and_unknown():
    git, files = Layer("get_current_branch", "list_files"), Layer("list_dir", "read_file")
    combined = CombinedTools(git, files)
    assert [t["function"]["name"] for t in combined.openai_tools] == [
        "get_current_branch", "list_files", "list_dir", "read_file"
    ]
    assert run(combined.call("read_file", None)).text == "read_file"
    assert git.calls == [] and files.calls == ["read_file"]
    assert run(combined.call("get_diff", None)).is_error


def test_combined_tools_name_collision_is_error():
    with pytest.raises(ValueError):
        CombinedTools(Layer("a"), Layer("a"))


def test_tool_names_are_names_of_model_tools_only_and_a_copy():
    tools, _, _ = make_tools()
    names = tools.tool_names
    assert names == [t["function"]["name"] for t in tools.openai_tools]
    assert "apply_change" not in names and "discard_change" not in names
    names.append("посторонний")
    assert "посторонний" not in tools.tool_names


def test_tool_names_of_empty_facade_is_empty_list():
    class Bare:
        openai_tools = []

    tools = FilesTools(
        Bare(),
        max_result_chars=100,
        registry=proposals.ProposalRegistry(600),
        chat_id=1,
        user_id=1,
    )
    assert tools.tool_names == []
