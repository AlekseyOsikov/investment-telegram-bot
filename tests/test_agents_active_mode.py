"""Тесты трекера активного режима чата (agents/active_mode.py): отметка режима на чат, смена,
снятие, независимость чатов. Состояние — словарь в памяти процесса, поэтому каждый тест
очищает свои чаты."""

import pytest

from agents import active_mode as am

CHATS = (101, 102)


@pytest.fixture(autouse=True)
def clean_chats():
    for chat_id in CHATS:
        am.clear_active_mode(chat_id)
    yield
    for chat_id in CHATS:
        am.clear_active_mode(chat_id)


def test_chat_without_mode_has_none():
    assert am.get_active_mode(101) is None


def test_set_mode_is_returned_by_get():
    am.set_active_mode(101, am.AGENT_MODE)
    assert am.get_active_mode(101) == "agent"


def test_set_mode_again_replaces_previous_one():
    am.set_active_mode(101, am.AGENT_MODE)
    am.set_active_mode(101, am.SMART_AGENT_MODE)
    assert am.get_active_mode(101) == "smart_agent"


def test_clear_removes_only_that_chat():
    am.set_active_mode(101, am.COMPARE_MODE)
    am.set_active_mode(102, am.SUPPORT_MODE)
    am.clear_active_mode(101)
    assert am.get_active_mode(101) is None
    assert am.get_active_mode(102) == "support"


def test_clear_of_chat_without_mode_is_not_an_error():
    am.clear_active_mode(101)
    am.clear_active_mode(101)
    assert am.get_active_mode(101) is None


def test_modes_are_distinct_strings():
    modes = [am.AGENT_MODE, am.COMPARE_MODE, am.SMART_AGENT_MODE, am.SUPPORT_MODE]
    assert len(set(modes)) == 4
    assert all(isinstance(mode, str) and mode for mode in modes)
