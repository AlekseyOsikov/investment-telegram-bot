"""Тесты правил режима поддержки (support_assistant/rules.py): команда, предупреждения, отсутствие
оценок и гарантий в фиксированных текстах."""

import re

from support_assistant import rules

FIXED_TEXTS = [
    rules.EXIT_TEXT,
    rules.EMPTY_QUESTION_TEXT,
    rules.INDEX_MISSING_TEXT,
    rules.SEARCH_FAILED_TEXT,
    rules.NOT_FOUND_TEXT,
    rules.CRM_UNAVAILABLE_WARNING,
]
FORBIDDEN = re.compile(
    r"гарант|точно вырастет|рекоменд|купи|продай|безрисков|доходност", re.IGNORECASE
)


def test_command_name():
    assert rules.ENTER_COMMAND == "support"


def test_enter_text_warns_about_sensitive_data_and_explains_exit():
    text = rules.ENTER_TEXT
    assert "номера счетов" in text and "карт" in text
    assert "паспортные" in text
    assert "/cancel" in text
    assert "инвестицион" in text  # режим не инвестиционный


def test_fixed_texts_have_no_assessments_or_guarantees():
    for text in FIXED_TEXTS + [rules.ENTER_TEXT]:
        assert not FORBIDDEN.search(text.replace("инвестиционном режиме", "")), text


def test_fixed_texts_are_non_empty_russian():
    for text in FIXED_TEXTS:
        assert re.search(r"[А-Яа-я]", text)


def test_unavailable_crm_warning_does_not_name_ticket_data():
    assert "недоступны" in rules.CRM_UNAVAILABLE_WARNING


def test_index_missing_text_points_to_operator_command():
    assert "make support-index" in rules.INDEX_MISSING_TEXT


def test_ticket_limits_are_positive():
    assert rules.TICKET_FIELD_MAX_CHARS > len(rules.TRUNCATION_MARK)
    assert rules.TICKET_EVENTS_MAX >= 1
