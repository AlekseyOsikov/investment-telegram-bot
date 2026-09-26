"""Тесты разбора аргументов /watch: чистые функции, без Telegram и сервера."""

import pytest

from price_watch.parse import ParseError, WatchRequest, normalize_interval, parse_watch_args


def _words(text):
    """Как PTB делит аргументы команды: по пробельным символам."""
    return text.split()


# --- нормализация периода ---


@pytest.mark.parametrize(
    "word, expected",
    [
        ("15m", "15m"),
        ("1h", "1h"),
        ("1d", "1d"),
        ("15м", "15m"),
        ("1ч", "1h"),
        ("2д", "2d"),
        ("1H", "1h"),
        ("1D", "1d"),
        ("30М", "30m"),
        ("007m", "007m"),
    ],
)
def test_interval_units_are_normalized_to_latin(word, expected):
    assert normalize_interval(word) == expected


@pytest.mark.parametrize("word", ["15", "m", "15 m", "18:00", "каждый", "1.5h", "-5m", "15mm", "SBER", ""])
def test_non_interval_words_are_rejected(word):
    assert normalize_interval(word) is None


# --- успешный разбор ---


def test_tickers_and_two_intervals():
    request = parse_watch_args(_words("SBER GAZP 15m 1h"))
    assert request == WatchRequest(secids=["SBER", "GAZP"], poll_interval="15m", report_interval="1h")


def test_russian_units_and_commas():
    request = parse_watch_args(_words("sber, gazp 15м 1ч"))
    assert request == WatchRequest(secids=["sber", "gazp"], poll_interval="15m", report_interval="1h")


def test_commas_without_spaces():
    assert parse_watch_args(_words("SBER,GAZP,LKOH 5m 1d")).secids == ["SBER", "GAZP", "LKOH"]


def test_single_ticker():
    assert parse_watch_args(_words("IMOEX 1h 1d")) == WatchRequest(["IMOEX"], "1h", "1d")


def test_repeated_and_leading_commas_are_ignored():
    assert parse_watch_args(_words(",SBER,, GAZP, 15m 1h")).secids == ["SBER", "GAZP"]


def test_duplicates_are_left_to_the_server():
    # Убирает повторы и приводит регистр сервер: бот проверяет только форму.
    assert parse_watch_args(_words("SBER sber 15m 1h")).secids == ["SBER", "sber"]


def test_bounds_are_not_checked_by_the_bot():
    # 1m ниже минимума и сводка чаще опроса: это решает сервер и говорит причину сам.
    request = parse_watch_args(_words("SBER 1m 1m"))
    assert isinstance(request, WatchRequest)
    assert request.poll_interval == "1m"


# --- ошибки формы ---


def test_empty_arguments_give_usage_only():
    assert parse_watch_args([]) == ParseError(None)
    assert parse_watch_args(["  "]) == ParseError(None)


@pytest.mark.parametrize("text", ["SBER 15m", "SBER GAZP", "SBER", "15m"])
def test_missing_interval_is_a_form_error(text):
    result = parse_watch_args(_words(text))
    assert isinstance(result, ParseError)
    assert result.reason


def test_no_tickers_before_intervals():
    result = parse_watch_args(_words("15m 1h"))
    assert isinstance(result, ParseError)
    assert "тикер" in result.reason


def test_interval_not_at_the_end():
    result = parse_watch_args(_words("SBER 15m GAZP 1h"))
    assert isinstance(result, ParseError)


def test_interval_among_tickers_is_a_form_error():
    result = parse_watch_args(_words("SBER 15m 1h 2h"))
    assert isinstance(result, ParseError)
    assert "15m" in result.reason


def test_interval_with_a_space_is_a_form_error():
    assert isinstance(parse_watch_args(_words("SBER 15 m 1 h")), ParseError)


@pytest.mark.parametrize("text", ["SBER 18:00 1h", "SBER каждый час", "SBER 1.5h 1d"])
def test_time_of_day_and_prose_are_form_errors(text):
    assert isinstance(parse_watch_args(_words(text)), ParseError)


@pytest.mark.parametrize("ticker", ["НЕТТАКОГО", "SB ER", "SBER!", "SBER/", "x" * 41, "СБЕР"])
def test_malformed_ticker_is_a_form_error_naming_the_ticker(ticker):
    result = parse_watch_args([ticker, "15m", "1h"])
    assert isinstance(result, ParseError)
    assert ticker in result.reason
    assert "тикер" in result.reason


@pytest.mark.parametrize("ticker", ["SBER", "SU26238RMFS4", "TMOS", "IMOEX", "RU000A0JX0J2", "BRK.B", "A_B-C", "x" * 40])
def test_wellformed_tickers_pass(ticker):
    assert parse_watch_args([ticker, "15m", "1h"]).secids == [ticker]
