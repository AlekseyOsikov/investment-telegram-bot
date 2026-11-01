"""Тесты разбора свободного текста /watch моделью-парсером: чистые функции без сети и Telegram."""

import pytest

from price_watch.nl_parse import (
    ERROR_CODES,
    FEWSHOT,
    SYSTEM_PROMPT,
    TICKERS,
    apply_guard,
    build_messages,
    duplicate_ticker_success,
    extract_json,
    native_chat_url,
    to_watch_request,
    validate_output,
)
from price_watch.parse import WatchRequest
from price_watch.texts import USAGE_TEXT, parser_error_text

# --- перенос из finetuning не изменился ---


def test_error_codes_unchanged():
    assert ERROR_CODES == (
        "empty",
        "not_watch",
        "missing_tickers",
        "missing_intervals",
        "missing_poll_interval",
        "missing_report_interval",
        "bad_interval",
        "bad_ticker",
        "ambiguous",
    )


def test_tickers_are_the_twenty_known():
    assert len(TICKERS) == 20
    assert TICKERS == sorted(TICKERS)


@pytest.mark.parametrize("ticker", ["SBER", "GAZP", "LKOH", "MOEX", "YDEX", "IRAO"])
def test_system_prompt_lists_tickers(ticker):
    assert ticker in SYSTEM_PROMPT


def test_system_prompt_mentions_error_and_format():
    assert "secids" in SYSTEM_PROMPT
    assert "poll_interval" in SYSTEM_PROMPT
    assert "report_interval" in SYSTEM_PROMPT


# --- validate_output ---


def test_valid_success_passes_validation():
    obj = {"ok": True, "secids": ["GAZP", "SBER"], "poll_interval": "15m", "report_interval": "1h"}
    assert validate_output(obj) == []


def test_valid_error_passes_validation():
    assert validate_output({"ok": False, "error": "missing_report_interval"}) == []


def test_unsorted_or_unknown_tickers_are_flagged():
    assert validate_output(
        {"ok": True, "secids": ["SBER", "GAZP"], "poll_interval": "15m", "report_interval": "1h"}
    )
    assert validate_output(
        {"ok": True, "secids": ["TSLA"], "poll_interval": "15m", "report_interval": "1h"}
    )


# --- guard ---


def test_guard_empty_input():
    assert apply_guard(
        "", {"ok": True, "secids": [], "poll_interval": "15m", "report_interval": "1h"}
    ) == {"ok": False, "error": "empty"}


def test_guard_missing_tickers():
    result = apply_guard("следи каждые 15 минут", {"ok": False, "error": "missing_tickers"})
    assert result == {"ok": False, "error": "missing_tickers"}


def test_guard_missing_intervals():
    result = apply_guard("следи за Сбером", {"ok": False, "error": "missing_intervals"})
    assert result == {"ok": False, "error": "missing_intervals"}


def test_guard_missing_report_interval_for_single_unnamed_period():
    result = apply_guard(
        "следи за Сбером каждые 15 минут", {"ok": False, "error": "missing_report_interval"}
    )
    assert result == {"ok": False, "error": "missing_report_interval"}


def test_guard_corrects_missing_and_extra_tickers():
    # Модель вернула только SBER, а в просьбе SBER и GAZP — guard дополняет и сортирует.
    output = {"ok": True, "secids": ["SBER"], "poll_interval": "15m", "report_interval": "1h"}
    result = apply_guard("следи за SBER и GAZP каждые 15 минут, сводку раз в час", output)
    assert result == {
        "ok": True,
        "secids": ["GAZP", "SBER"],
        "poll_interval": "15m",
        "report_interval": "1h",
    }


def test_guard_fixes_wrong_ticker_alias():
    # «Магнитом» — это MGNT, а не MTSS.
    output = {"ok": True, "secids": ["MTSS"], "poll_interval": "15m", "report_interval": "1h"}
    result = apply_guard("следи за Магнитом каждые 15 минут, сводку раз в час", output)
    assert result["secids"] == ["MGNT"]


def test_duplicate_ticker_ambiguous_becomes_success():
    output = {"ok": False, "error": "ambiguous"}
    result = duplicate_ticker_success(
        "следи за SBER и SBER каждые 15 минут, сводку раз в час", output
    )
    assert result["ok"] is True
    assert result["secids"] == ["SBER"]
    assert result["poll_interval"] == "15m"
    assert result["report_interval"] == "1h"


# --- build_messages / extract_json / to_watch_request ---


def test_build_messages_system_fewshot_user():
    messages = build_messages("Следи за Сбером каждые 15 минут, сводку раз в час")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    # два few-shot (user+assistant) + вопрос пользователя = 6 сообщений
    assert len(messages) == 1 + 2 * len(FEWSHOT) + 1
    assert messages[-1] == {
        "role": "user",
        "content": "Следи за Сбером каждые 15 минут, сводку раз в час",
    }
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_extract_json_plain():
    assert extract_json('{"ok":false,"error":"empty"}') == {"ok": False, "error": "empty"}


def test_extract_json_from_code_fence():
    assert extract_json('```json\n{"ok":false,"error":"empty"}\n```') == {
        "ok": False,
        "error": "empty",
    }


def test_extract_json_from_surrounding_text():
    assert extract_json('вот: {"ok":true,"secids":["SBER"]} конец')["ok"] is True


def test_extract_json_rejects_non_json():
    assert extract_json("не json") is None


def test_to_watch_request_success():
    obj = {"ok": True, "secids": ["SBER"], "poll_interval": "15m", "report_interval": "1h"}
    assert to_watch_request(obj) == WatchRequest(["SBER"], "15m", "1h")


def test_to_watch_request_error():
    assert to_watch_request({"ok": False, "error": "missing_report_interval"}) == (
        "missing_report_interval"
    )


def test_to_watch_request_malformed_success_is_error():
    assert to_watch_request({"ok": True, "secids": []}) == "empty"


# --- native_chat_url ---


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://localhost:11434/v1", "http://localhost:11434/api/chat"),
        ("http://localhost:11434/v1/", "http://localhost:11434/api/chat"),
        ("http://localhost:11434", "http://localhost:11434/api/chat"),
    ],
)
def test_native_chat_url(base_url, expected):
    assert native_chat_url(base_url) == expected


# --- тексты ошибок ---


def test_parser_error_text_known_code():
    text = parser_error_text("missing_report_interval")
    assert "сводк" in text
    assert USAGE_TEXT in text


def test_parser_error_text_unknown_code_falls_back():
    text = parser_error_text("что-то-неведомое")
    assert "Не удалось разобрать" in text
    assert USAGE_TEXT in text
