"""Тесты обращений к серверу новостей (news_digest/news_session.py): только чистые функции, без
процессов и сети."""

import os
import sys

import pytest
from mcp.types import CallToolResult, TextContent

from mcp_integration.watch_session import WatchProtocolError
from news_digest import news_session as ns


def test_server_params_use_same_interpreter_and_no_shell():
    params = ns.build_news_server_params(
        "/var/bot/data/news.sqlite3", ["https://a.example/rss", "https://b.example/x.json"], 7
    )
    assert params.command == sys.executable
    assert params.args[0] == ns.SERVER_SCRIPT and os.path.isfile(ns.SERVER_SCRIPT)
    assert params.args[1:5] == ["--db", "/var/bot/data/news.sqlite3", "--retention-days", "7"]
    assert params.args[5:] == ["--feeds", "https://a.example/rss", "https://b.example/x.json"]


def test_hostile_values_stay_single_arguments():
    db = "/tmp/a b/$(id).sqlite3"
    feed = "https://x.example/rss; rm -rf ~"
    params = ns.build_news_server_params(db, [feed], 3)
    assert params.args[2] == db and params.args[-1] == feed
    assert len(params.args) == 7


def test_no_feeds_means_no_feeds_flag():
    assert "--feeds" not in ns.build_news_server_params("/tmp/n.sqlite3", [], 7).args


def test_friendly_errors_in_russian():
    assert ns.friendly_error(ns.NewsToolError("Лимит исчерпан")) == "Лимит исчерпан"
    assert "запуска" in ns.friendly_error(FileNotFoundError())
    assert "не ответил" in ns.friendly_error(TimeoutError())
    assert "неожиданный" in ns.friendly_error(WatchProtocolError("x"))
    assert "не удалось соединиться" in ns.friendly_error(RuntimeError("boom"))


def test_unwrap_exception_group_is_shared_with_watch_session():
    class Group(Exception):
        def __init__(self, *exceptions):
            self.exceptions = exceptions

    inner = TimeoutError("t")
    from mcp_integration.watch_session import unwrap_exception_group

    assert unwrap_exception_group(Group(Group(inner))) is inner


def test_tool_error_result_becomes_news_tool_error(monkeypatch):
    # результат isError с префиксом SDK превращается в NewsToolError с чистым русским текстом
    from mcp_integration.watch_session import WatchToolError, result_to_dict

    result = CallToolResult(
        content=[TextContent(type="text", text="Error executing tool news_ack: Плохой запрос.")],
        is_error=True,
    )
    with pytest.raises(WatchToolError) as caught:
        result_to_dict(result)
    assert str(caught.value) == "Плохой запрос."
