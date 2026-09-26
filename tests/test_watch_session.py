"""Тесты обращений к серверу расписания: только чистые функции.

Ни сети, ни запуска процессов: на входе настоящие типы MCP SDK (mcp.types), собранные
вручную. Сам запуск сервера и живой обмен проверяются вручную (см. tasks.md изменения
add-price-watch-commands, задачи 3.2 и 8.2).
"""

import os

import pytest
from mcp.types import CallToolResult, TextContent

import config
from mcp_integration.watch_session import (
    WatchProtocolError,
    WatchToolError,
    build_watch_server_params,
    result_to_dict,
    unwrap_exception_group,
)


def _result(text="", structured=None, is_error=False):
    return CallToolResult(
        content=[TextContent(type="text", text=text)] if text else [],
        structured_content=structured,
        is_error=is_error,
    )


# --- параметры запуска ---


def test_server_params_enable_watch_mode_with_absolute_db_path():
    params = build_watch_server_params("/opt/mcp-moex", "/var/bot/data/watch.sqlite3")
    assert params.command == "uv"
    assert params.args == [
        "run", "--directory", "/opt/mcp-moex", "mcp-moex", "--watch-db", "/var/bot/data/watch.sqlite3",
    ]


def test_hostile_values_stay_single_arguments():
    directory = "/opt/x; rm -rf ~ && echo 'hi'"
    db_path = "/tmp/a b/$(id).sqlite3"
    params = build_watch_server_params(directory, db_path)
    assert params.args[2] == directory
    assert params.args[-1] == db_path
    assert len(params.args) == 6


def test_configured_db_path_is_absolute_even_for_relative_setting():
    # uv run --directory меняет рабочий каталог сервера: относительный путь указал бы
    # внутрь каталога mcp-moex, поэтому config делает его абсолютным при загрузке.
    assert os.path.isabs(config.PRICE_WATCH_DB)


# --- разбор результата ---


def test_structured_content_is_returned_as_is():
    assert result_to_dict(_result(structured={"active": False})) == {"active": False}


def test_json_text_is_used_when_there_is_no_structured_content():
    assert result_to_dict(_result(text='{"stopped": true}')) == {"stopped": True}


def test_server_error_keeps_the_server_text():
    text = "Бумага с тикером «XXXX» не найдена на Московской бирже."
    with pytest.raises(WatchToolError) as info:
        result_to_dict(_result(text=text, is_error=True))
    assert str(info.value) == text


def test_sdk_error_prefix_is_stripped_from_the_server_text():
    raw = "Error executing tool watch_set: Параметр poll_interval: «1m» вне границ."
    with pytest.raises(WatchToolError) as info:
        result_to_dict(_result(text=raw, is_error=True))
    assert str(info.value) == "Параметр poll_interval: «1m» вне границ."


def test_error_that_is_only_the_prefix_still_says_something():
    with pytest.raises(WatchToolError) as info:
        result_to_dict(_result(text="Error executing tool watch_stop:", is_error=True))
    assert str(info.value)


def test_server_error_ignores_structured_content():
    with pytest.raises(WatchToolError):
        result_to_dict(_result(text="Ошибка", structured={"ok": True}, is_error=True))


def test_server_error_without_text_still_says_something():
    with pytest.raises(WatchToolError) as info:
        result_to_dict(_result(is_error=True))
    assert str(info.value)


def test_empty_result_is_a_protocol_error():
    with pytest.raises(WatchProtocolError):
        result_to_dict(_result())


def test_non_json_text_is_a_protocol_error():
    with pytest.raises(WatchProtocolError):
        result_to_dict(_result(text="не JSON"))


def test_json_text_that_is_not_an_object_is_a_protocol_error():
    with pytest.raises(WatchProtocolError):
        result_to_dict(_result(text="[1, 2]"))


# --- группы исключений ---


def test_single_exception_group_is_unwrapped_to_the_leaf():
    leaf = TimeoutError("late")
    assert unwrap_exception_group(ExceptionGroup("g", [leaf])) is leaf


def test_nested_single_exception_groups_are_unwrapped():
    leaf = FileNotFoundError("uv")
    assert unwrap_exception_group(ExceptionGroup("a", [ExceptionGroup("b", [leaf])])) is leaf


def test_group_of_several_exceptions_stays_a_group():
    group = ExceptionGroup("g", [ValueError("1"), TypeError("2")])
    assert unwrap_exception_group(group) is group


def test_plain_exception_is_returned_unchanged():
    error = RuntimeError("x")
    assert unwrap_exception_group(error) is error
