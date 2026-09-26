"""Тесты текстов команд опроса цен на образцах ответов сервера (контракт mcp-moex)."""

from price_watch import texts

SET_RESULT = {
    "chat_id": 42,
    "secids": ["SBER", "GAZP"],
    "poll_interval": "15m",
    "report_interval": "1h",
    "replaced": False,
    "next_poll_at": "2026-09-26T12:15:00+03:00",
    "next_report_at": "2026-09-26T13:00:00+03:00",
    "first_samples": [
        {"secid": "SBER", "price": 301.2, "price_unit": "RUB", "quote_at": "2026-09-26T11:45:00+03:00"},
        {"secid": "GAZP", "price": 120.0, "price_unit": "RUB", "quote_at": "2026-09-26T11:45:00+03:00"},
    ],
}


# --- время и склонения ---


def test_format_msk_keeps_moscow_time():
    assert texts.format_msk("2026-09-26T12:15:00+03:00") == "26.09 12:15 (МСК)"


def test_format_msk_converts_other_offsets_to_moscow():
    assert texts.format_msk("2026-09-26T09:15:00+00:00") == "26.09 12:15 (МСК)"


def test_format_msk_without_zone_suffix():
    assert texts.format_msk("2026-09-26T12:15:00+03:00", with_zone=False) == "26.09 12:15"


def test_format_msk_garbage_is_returned_as_is_and_none_is_a_dash():
    assert texts.format_msk("завтра") == "завтра"
    assert texts.format_msk(None) == "—"


def test_every_text_declines_units():
    assert texts.every_text("5m") == "раз в 5 минут"
    assert texts.every_text("1m") == "раз в минуту"
    assert texts.every_text("21m") == "раз в 21 минуту"
    assert texts.every_text("22m") == "раз в 22 минуты"
    assert texts.every_text("11m") == "раз в 11 минут"
    assert texts.every_text("1h") == "раз в час"
    assert texts.every_text("2h") == "раз в 2 часа"
    assert texts.every_text("12h") == "раз в 12 часов"
    assert texts.every_text("1d") == "раз в день"
    assert texts.every_text("3d") == "раз в 3 дня"
    assert texts.every_text("7d") == "раз в 7 дней"


def test_every_text_leaves_unknown_value_as_is():
    assert texts.every_text("часто") == "часто"


def test_format_price_uses_comma_and_trims_zeros():
    assert texts.format_price(301.2) == "301,2"
    assert texts.format_price(100.0) == "100"
    assert texts.format_price(96.4375) == "96,4375"
    assert texts.format_price(0.5) == "0,5"


# --- /watch ---


def test_watch_set_text_has_tickers_periods_terms_prices_and_delay_note():
    text = texts.watch_set_text(SET_RESULT)
    assert "Опрос задан" in text
    assert "заменён" not in text
    assert "SBER, GAZP" in text
    assert "раз в 15 минут" in text and "раз в час" in text
    assert "26.09 12:15 (МСК)" in text and "26.09 13:00 (МСК)" in text
    assert "SBER: 301,2 RUB" in text
    assert "15 минут" in text and "индексы без задержки" in text


def test_watch_set_text_marks_replacement():
    assert "прежний опрос заменён" in texts.watch_set_text({**SET_RESULT, "replaced": True})


def test_watch_set_text_shows_bond_and_index_units():
    result = {
        **SET_RESULT,
        "secids": ["SU26238RMFS4", "IMOEX"],
        "first_samples": [
            {"secid": "SU26238RMFS4", "price": 51.79, "price_unit": "percent_of_face", "quote_at": "2026-09-26T11:45:00+03:00"},
            {"secid": "IMOEX", "price": 2801.5, "price_unit": "points", "quote_at": "2026-09-26T11:59:00+03:00"},
        ],
    }
    text = texts.watch_set_text(result)
    assert "51,79 % от номинала" in text
    assert "2801,5 пунктов" in text


def test_watch_set_text_without_samples_still_works():
    text = texts.watch_set_text({**SET_RESULT, "first_samples": []})
    assert "Первые цены" not in text


# --- /watch_status ---

STATUS_ACTIVE = {
    "active": True,
    "chat_id": 42,
    "secids": ["SBER", "GAZP"],
    "poll_interval": "15m",
    "report_interval": "1h",
    "started_at": "2026-09-26T12:00:00+03:00",
    "next_poll_at": "2026-09-26T12:15:00+03:00",
    "next_report_at": "2026-09-26T13:00:00+03:00",
    "samples_in_period": 4,
    "pending_reports": 0,
}


def test_status_text_for_active_watch():
    text = texts.watch_status_text(STATUS_ACTIVE)
    assert "SBER, GAZP" in text
    assert "Задан: 26.09 12:00 (МСК)" in text
    assert "Замеров в текущем периоде: 4" in text
    assert "ожидающих доставки" not in text


def test_status_text_shows_pending_reports_only_when_there_are_some():
    assert "ожидающих доставки: 2" in texts.watch_status_text({**STATUS_ACTIVE, "pending_reports": 2})


def test_status_text_without_watch_points_to_the_command():
    text = texts.watch_status_text({"active": False})
    assert "/watch" in text
    assert text == texts.NO_WATCH_TEXT


# --- /watch_stop и подсказка ---


def test_stop_texts_differ_for_existing_and_missing_watch():
    stopped = texts.watch_stop_text({"chat_id": 42, "stopped": True})
    missing = texts.watch_stop_text({"chat_id": 42, "stopped": False})
    assert stopped != missing
    assert "остановлен" in stopped
    assert "не было" in missing


def test_usage_text_has_format_example_and_units():
    usage = texts.USAGE_TEXT
    assert "/watch <тикеры> <период опроса> <период сводки>" in usage
    assert "/watch SBER GAZP 15m 1h" in usage
    assert "m или м" in usage and "h или ч" in usage and "d или д" in usage


# --- отказы сервера ---


def test_russian_server_refusal_passes_through_unchanged():
    text = "Параметр poll_interval: «1m» вне границ. Период опроса от 5m до 1d включительно."
    assert texts.friendly_tool_error(text) == text


def test_raw_schema_validation_dump_is_replaced_with_a_russian_hint():
    dump = "1 validation error for watch_setArguments\nsecids.0\n  String should match pattern"
    friendly = texts.friendly_tool_error(dump)
    assert "validation" not in friendly
    assert "/watch SBER GAZP 15m 1h" in friendly
