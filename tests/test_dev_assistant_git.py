"""Тесты git-слоя (dev_assistant/git_session.py): схема без repo_path, подстановка пути,
перечень инструментов. Без процессов: сессия подменена фейком."""

import asyncio
import json

from dev_assistant import git_session
from dev_assistant.git_session import GIT_TOOL_ALLOWLIST, GitTools
from mcp_integration.market_session import ToolOutcome

ROOT = "/srv/project"


def tool(name, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


REPO = {"type": "string"}


class FakeInner:
    def __init__(self, names=("get_current_branch", "list_files", "get_diff", "list_roots")):
        self.openai_tools = [
            tool(n, {"repo_path": REPO, "ref": {"type": "string"}}, ["repo_path"]) for n in names
        ]
        self.calls: list[tuple[str, str | None]] = []

    async def call(self, name, raw):
        self.calls.append((name, raw))
        return ToolOutcome("ok")


def run(coro):
    return asyncio.run(coro)


def test_allowlist_is_exactly_branch_and_files():
    assert GIT_TOOL_ALLOWLIST == {"get_current_branch", "list_files"}


def test_server_params_use_root_and_no_shell():
    params = git_session.build_git_server_params("/opt/mcp-git", ROOT)
    assert params.command == "uv"
    assert params.args == ["run", "--directory", "/opt/mcp-git", "mcp-git", "--root", ROOT]


def test_strip_repo_path_removes_property_and_required_without_mutating_source():
    source = tool("list_files", {"repo_path": REPO, "path": {"type": "string"}}, ["repo_path"])
    stripped = git_session.strip_repo_path(source)
    params = stripped["function"]["parameters"]
    assert "repo_path" not in params["properties"] and "path" in params["properties"]
    assert "required" not in params
    assert "repo_path" in source["function"]["parameters"]["properties"]


def test_facade_exposes_only_allowlisted_tools_without_repo_path():
    facade = GitTools(FakeInner(), ROOT)
    assert facade.tool_names == ["get_current_branch", "list_files"]
    for item in facade.openai_tools:
        assert "repo_path" not in item["function"]["parameters"]["properties"]


def test_call_overwrites_foreign_repo_path():
    inner = FakeInner()
    facade = GitTools(inner, ROOT)
    outcome = run(facade.call("list_files", json.dumps({"repo_path": "/etc", "ref": "main"})))
    assert not outcome.is_error
    name, raw = inner.calls[0]
    assert name == "list_files"
    assert json.loads(raw) == {"repo_path": ROOT, "ref": "main"}


def test_call_without_arguments_gets_repo_path():
    inner = FakeInner()
    run(GitTools(inner, ROOT).call("get_current_branch", None))
    assert json.loads(inner.calls[0][1]) == {"repo_path": ROOT}


def test_invalid_json_is_error_for_model_and_not_forwarded():
    inner = FakeInner()
    outcome = run(GitTools(inner, ROOT).call("list_files", "{не json"))
    assert outcome.is_error and inner.calls == []


def test_non_object_arguments_rejected():
    inner = FakeInner()
    outcome = run(GitTools(inner, ROOT).call("list_files", "[1,2]"))
    assert outcome.is_error and inner.calls == []


def test_diff_and_roots_are_rejected_by_name_without_server_call():
    inner = FakeInner()
    facade = GitTools(inner, ROOT)
    for name in ("get_diff", "list_roots", "unknown"):
        outcome = run(facade.call(name, "{}"))
        assert outcome.is_error and "нет среди доступных" in outcome.text
    assert inner.calls == []


def test_server_without_allowed_tools_gives_empty_facade():
    facade = GitTools(FakeInner(names=("get_diff",)), ROOT)
    assert facade.openai_tools == [] and facade.tool_names == []


def test_display_arguments_hides_repo_path():
    raw = '{"repo_path": "/etc", "ref": "main"}'
    assert git_session.display_arguments(raw) == '{"ref": "main"}'
    assert git_session.display_arguments('{"repo_path": "/etc"}') == ""
    assert git_session.display_arguments(None) == ""
    assert git_session.display_arguments("{не json") == "{не json"
