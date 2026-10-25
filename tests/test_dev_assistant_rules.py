"""Тесты чистых правил режима ассистента разработчика: разбор списка id и проверка доступа."""

from types import SimpleNamespace

import pytest

from dev_assistant import rules


def test_empty_values_disable_mode():
    assert rules.parse_user_ids("") == ()
    assert rules.parse_user_ids("   ") == ()
    assert rules.parse_user_ids(None) == ()
    assert rules.parse_user_ids(" , ,") == ()


def test_parses_ids_with_spaces_and_keeps_order():
    assert rules.parse_user_ids(" 42 , 7,100 ") == (42, 7, 100)


def test_duplicates_collapse():
    assert rules.parse_user_ids("5,7,5,7") == (5, 7)


@pytest.mark.parametrize("raw", ["abc", "1,two,3", "1.5", "12abc", "0", "-100500", "1,-2"])
def test_invalid_elements_raise(raw):
    with pytest.raises(ValueError):
        rules.parse_user_ids(raw)


def test_error_names_element():
    with pytest.raises(ValueError, match="two"):
        rules.parse_user_ids("1,two")


def test_format_user_ids_joins_and_collapses_duplicates():
    assert rules.format_user_ids([1, 2, 3]) == "1,2,3"
    assert rules.format_user_ids([5, 7, 5, 7]) == "5,7"
    assert rules.format_user_ids([]) == ""


@pytest.mark.parametrize("ids", [[], [42], [3, 1, 2], [5, 7, 5, 7, 9]])
def test_format_user_ids_roundtrip(ids):
    assert rules.parse_user_ids(rules.format_user_ids(ids)) == tuple(dict.fromkeys(ids))


def test_authorized_by_user_id():
    assert rules.is_authorized(SimpleNamespace(id=7), (5, 7))
    assert not rules.is_authorized(SimpleNamespace(id=8), (5, 7))


def test_none_sender_is_rejected():
    assert not rules.is_authorized(None, (5, 7))
    assert not rules.is_authorized(SimpleNamespace(id=None), (5, 7))


def test_empty_list_rejects_everyone():
    assert not rules.is_authorized(SimpleNamespace(id=7), ())


def test_bool_id_is_not_accepted():
    assert not rules.is_authorized(SimpleNamespace(id=True), (1,))
