"""Тесты инвариантов smart-агента (agents/invariants.py): разбор ввода, чтение из файла памяти,
нумерованный список, системное сообщение для основного вызова, запрос ревизору и разбор его
ответа, текст нарушений.

Без сети, Telegram и модели: ответ ревизора представлен РАЗОБРАННЫМ словарём.
"""

import datetime as dt
import re

import pytest

from agents import invariants as inv

GUARANTEE = re.compile(r"гарантир|точно вырастет|рекомендуем купить|без риска", re.IGNORECASE)


def _items() -> list[dict]:
    return [
        {"text": "без криптовалют", "category": "instruments", "created_at": "t1"},
        {"text": "доля акций не выше 40%", "category": "risk", "created_at": "t2"},
        {"text": "ребалансировка раз в год", "category": "process", "created_at": "t3"},
    ]


# --- parse_input --------------------------------------------------------------------------


def test_parse_input_with_english_category():
    assert inv.parse_input("risk: доля акций не выше 40%") == (
        inv.CATEGORY_RISK,
        "доля акций не выше 40%",
    )


def test_parse_input_with_russian_category_and_spaces_and_case():
    assert inv.parse_input("  Инструменты :  без плеча ") == (
        inv.CATEGORY_INSTRUMENTS,
        "без плеча",
    )
    assert inv.parse_input("СТРАТЕГИИ: только индексы") == (
        inv.CATEGORY_STRATEGY,
        "только индексы",
    )


@pytest.mark.parametrize(
    ("alias", "category"),
    [
        ("strategy", inv.CATEGORY_STRATEGY),
        ("стратегия", inv.CATEGORY_STRATEGY),
        ("инструмент", inv.CATEGORY_INSTRUMENTS),
        ("риски", inv.CATEGORY_RISK),
        ("процесс", inv.CATEGORY_PROCESS),
        ("процессы", inv.CATEGORY_PROCESS),
        ("прочее", inv.CATEGORY_OTHER),
        ("other", inv.CATEGORY_OTHER),
    ],
)
def test_parse_input_accepts_all_aliases(alias, category):
    assert inv.parse_input(f"{alias}: текст") == (category, "текст")


def test_parse_input_without_colon_goes_to_other_category():
    assert inv.parse_input("  без плеча  ") == (inv.CATEGORY_OTHER, "без плеча")


def test_parse_input_with_unknown_category_keeps_whole_text():
    assert inv.parse_input("цвет: синий") == (inv.CATEGORY_OTHER, "цвет: синий")


def test_parse_input_with_category_but_empty_text_keeps_whole_text():
    assert inv.parse_input("risk:") == (inv.CATEGORY_OTHER, "risk:")
    assert inv.parse_input("risk:   ") == (inv.CATEGORY_OTHER, "risk:")


def test_parse_input_splits_only_on_the_first_colon():
    assert inv.parse_input("process: этапы: раз в год") == (
        inv.CATEGORY_PROCESS,
        "этапы: раз в год",
    )


def test_parse_input_of_empty_string():
    assert inv.parse_input("") == (inv.CATEGORY_OTHER, "")


# --- new_invariant ------------------------------------------------------------------------


class _FixedDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 1, 12, 0, 0)


def test_new_invariant_strips_text_and_stamps_time(monkeypatch):
    monkeypatch.setattr(inv, "datetime", _FixedDateTime)
    assert inv.new_invariant("  без плеча ", inv.CATEGORY_RISK) == {
        "text": "без плеча",
        "category": "risk",
        "created_at": "2026-10-01T12:00:00",
    }


def test_new_invariant_with_unknown_category_becomes_other():
    assert inv.new_invariant("текст", "нет такой")["category"] == inv.CATEGORY_OTHER


# --- sanitize / sanitize_list -------------------------------------------------------------


def test_sanitize_keeps_valid_item():
    item = {"text": " запрет ", "category": "risk", "created_at": "2026-01-01"}
    assert inv.sanitize(item) == {"text": "запрет", "category": "risk", "created_at": "2026-01-01"}


@pytest.mark.parametrize("item", [None, "строка", 5, [], {"text": ""}, {"text": "  "}, {}])
def test_sanitize_drops_junk_and_empty_text(item):
    assert inv.sanitize(item) is None


def test_sanitize_fixes_category_and_created_at():
    result = inv.sanitize({"text": "запрет", "category": "нет", "created_at": 5})
    assert result == {"text": "запрет", "category": "other", "created_at": ""}
    assert inv.sanitize({"text": "запрет"})["category"] == "other"


def test_sanitize_converts_non_string_text():
    assert inv.sanitize({"text": 40})["text"] == "40"


@pytest.mark.parametrize("raw", [None, "x", {"text": "a"}, 5])
def test_sanitize_list_of_non_list_is_empty(raw):
    assert inv.sanitize_list(raw) == []


def test_sanitize_list_filters_junk_and_keeps_order():
    raw = [{"text": "первый"}, None, {"text": ""}, "x", {"text": "второй", "category": "risk"}]
    assert [item["text"] for item in inv.sanitize_list(raw)] == ["первый", "второй"]


# --- render_numbered ----------------------------------------------------------------------


def test_render_numbered_has_numbers_from_one_with_category_labels():
    assert inv.render_numbered(_items()) == (
        "1. [инструменты] без криптовалют\n"
        "2. [риск] доля акций не выше 40%\n"
        "3. [процесс] ребалансировка раз в год"
    )


def test_render_numbered_of_empty_list_is_empty():
    assert inv.render_numbered([]) == ""


# --- build_context_message ----------------------------------------------------------------


def test_context_message_is_system_with_numbered_list_and_rules():
    message = inv.build_context_message(_items())
    assert message["role"] == "system"
    content = message["content"]
    assert inv.render_numbered(_items()) in content
    assert "/smart_agent_invariant_remove" in content
    for number in range(1, 6):
        assert f"\n{number}. " in content


def test_context_message_does_not_weaken_mandatory_warnings():
    content = inv.build_context_message(_items())["content"]
    assert "не отменяют и не ослабляют обязательные правила основной инструкции" in content
    assert "не индивидуальная рекомендация" in content
    assert "приоритет над предпочтениями профиля" in content


def test_context_message_forbids_announcing_the_check():
    content = inv.build_context_message(_items())["content"]
    assert "молча" in content
    assert "не описывай" in content


def test_context_message_has_no_guarantee_wording():
    assert GUARANTEE.search(inv.build_context_message(_items())["content"]) is None


# --- build_review_user_content ------------------------------------------------------------


def test_review_content_has_invariants_goal_and_result_but_no_dialog():
    text = inv.build_review_user_content(
        _items(), "Составление портфеля", "портфель на пенсию", {"структура": "акции 40%"}
    )
    assert inv.render_numbered(_items()) in text
    assert "Задача: Составление портфеля — «портфель на пенсию»." in text
    assert "- структура: акции 40%" in text


def test_review_content_lists_every_result_item_in_order():
    text = inv.build_review_user_content(_items(), "С", "г", {"a": "1", "b": "2"})
    assert text.index("- a: 1") < text.index("- b: 2")


# --- parse_review_response ----------------------------------------------------------------


def _violation(index, why="причина"):
    return {"index": index, "why": why}


def test_review_ok_verdict_gives_no_violations():
    parsed = {"verdict": "ok", "violations": [_violation(1)]}
    assert inv.parse_review_response(parsed, _items()) == []
    assert inv.parse_review_response({}, _items()) == []


def test_review_violation_copies_invariant_text_and_why():
    result = inv.parse_review_response(
        {"verdict": "violated", "violations": [_violation(2, " в структуре 50% акций ")]},
        _items(),
    )
    assert result == [
        {"index": 2, "text": "доля акций не выше 40%", "why": "в структуре 50% акций"}
    ]


def test_review_empty_why_gets_default_reason_and_long_why_is_cut():
    result = inv.parse_review_response(
        {"verdict": "violated", "violations": [_violation(1, "  "), _violation(2, "а" * 500)]},
        _items(),
    )
    assert result[0]["why"] == "результат противоречит этому ограничению"
    assert len(result[1]["why"]) == 200


def test_review_missing_why_key_gets_default_reason():
    result = inv.parse_review_response(
        {"verdict": "violated", "violations": [{"index": 1}]}, _items()
    )
    assert result[0]["why"] == "результат противоречит этому ограничению"


@pytest.mark.parametrize("violations", [None, "x", {"index": 1}, 5])
def test_review_non_list_violations_give_nothing(violations):
    parsed = {"verdict": "violated", "violations": violations}
    assert inv.parse_review_response(parsed, _items()) == []


def test_review_skips_junk_entries_bad_numbers_out_of_range_and_duplicates():
    raw = [
        "мусор",
        None,
        {"index": "x"},
        {"index": None},
        {"index": 0},
        {"index": 4},
        {"index": -1},
        {"index": "3", "why": "строковый номер"},
        {"index": 3, "why": "дубль"},
        {"index": 1.0, "why": "дробный номер"},
    ]
    result = inv.parse_review_response({"verdict": "violated", "violations": raw}, _items())
    assert [(v["index"], v["why"]) for v in result] == [
        (3, "строковый номер"),
        (1, "дробный номер"),
    ]


def test_review_violated_without_valid_numbers_means_no_violations():
    result = inv.parse_review_response(
        {"verdict": "violated", "violations": [_violation(9)]}, _items()
    )
    assert result == []
    assert inv.parse_review_response({"verdict": "violated"}, []) == []


# --- render_violations --------------------------------------------------------------------


def test_render_violations_one_line_per_claim():
    violations = [
        {"index": 1, "text": "без криптовалют", "why": "есть токены"},
        {"index": 2, "text": "доля акций не выше 40%", "why": "50%"},
    ]
    assert inv.render_violations(violations) == (
        "- инвариант №1 (без криптовалют): есть токены\n"
        "- инвариант №2 (доля акций не выше 40%): 50%"
    )
    assert inv.render_violations([]) == ""


def test_categories_have_labels_and_hints_for_user_facing_categories():
    assert set(inv.CATEGORY_HINTS) <= set(inv.CATEGORY_LABELS)
    assert inv.CATEGORY_OTHER in inv.CATEGORY_LABELS
