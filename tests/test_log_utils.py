"""Тесты безопасного представления текста для логов (src/log_utils.py)."""

from log_utils import preview_for_log


def test_plain_text_is_preserved() -> None:
    assert preview_for_log("Как купить акции?") == "Как купить акции?"


def test_email_is_masked() -> None:
    assert preview_for_log("пишите на user@example.com") == "пишите на [email]"


def test_phone_is_masked() -> None:
    assert preview_for_log("звоните +7 (999) 123-45-67") == "звоните [phone]"


def test_card_number_is_masked() -> None:
    assert preview_for_log("карта 4111 1111 1111 1111") == "карта [number]"


def test_newlines_and_control_chars_become_single_line() -> None:
    assert preview_for_log("a\nb\x00c") == "a b c"


def test_long_text_is_truncated_with_ellipsis() -> None:
    result = preview_for_log("x" * 600, limit=500)
    assert len(result) == 501 and result.endswith("…")


def test_empty_text_is_empty() -> None:
    assert preview_for_log("") == ""
    assert preview_for_log(None) == ""  # type: ignore[arg-type]
