"""Тесты строки о совпадении по времени с движением цены (news_digest/price_context.py)."""

import re
from datetime import datetime, timedelta, timezone

from news_digest import price_context as pc

MSK = timezone(timedelta(hours=3))
NOW = datetime(2026, 10, 16, 18, 0, tzinfo=MSK)


def candle(hour, close, open_=None, day=16):
    return {
        "begin": f"2026-10-{day:02d}T{hour:02d}:00:00+03:00",
        "open": open_ if open_ is not None else close,
        "high": close,
        "low": close,
        "close": close,
    }


CANDLES = [
    candle(10, 100.0),
    candle(11, 101.0),
    candle(12, 102.0),
    candle(13, 103.0),
    candle(14, 104.0),
    candle(15, 105.0),
]
PUBLISHED = datetime(2026, 10, 16, 11, 30, tzinfo=MSK)


def test_change_between_last_closed_candle_before_news_and_last_known_price():
    change = pc.price_change_since(CANDLES, PUBLISHED, NOW)
    assert change is not None
    assert (
        change.start_price == 100.0 and change.end_price == 105.0
    )  # свеча 10:00 закрылась в 11:00
    assert round(change.pct, 2) == 5.0
    assert change.start_at.hour == 11 and change.end_at.hour == 16


def test_no_data_gives_no_line():
    assert pc.price_change_since(None, PUBLISHED, NOW) is None
    assert pc.price_change_since([], PUBLISHED, NOW) is None
    assert pc.coincidence_line(None) is None


def test_news_before_first_candle_gives_no_line():
    assert pc.price_change_since(CANDLES, datetime(2026, 10, 16, 9, 0, tzinfo=MSK), NOW) is None


def test_too_short_interval_gives_no_line():
    published = datetime(2026, 10, 16, 17, 0, tzinfo=MSK)  # данных после новости ещё нет
    assert pc.price_change_since(CANDLES, published, NOW) is None
    assert pc.price_change_since(CANDLES, PUBLISHED, NOW, min_span_minutes=600) is None


def test_old_news_and_future_news_give_no_line():
    old = datetime(2026, 10, 14, 11, 30, tzinfo=MSK)
    assert pc.price_change_since(CANDLES, old, NOW) is None
    assert pc.price_change_since(CANDLES, NOW + timedelta(hours=1), NOW) is None


def test_broken_candles_are_skipped():
    broken = [{"begin": "мусор", "close": 1}, {"begin": "2026-10-16T10:00:00+03:00"}, *CANDLES]
    assert pc.price_change_since(broken, PUBLISHED, NOW) is not None


def test_line_wording_has_caveats_and_no_cause_words():
    line = pc.coincidence_line(pc.price_change_since(CANDLES, PUBLISHED, NOW))
    assert "Цена в это же время" in line and "+5%" in line
    assert "задержаны" in line and "не доходность" in line
    assert "связь с новостью не установлена" in line
    cause = re.compile(
        r"из-за|вследствие|в результате|вызвал\w*|причин\w*(?! не)|обусловлен\w*|"
        r"под влиянием|благодаря|отреагировал\w*|среагировал\w*",
        re.IGNORECASE,
    )
    assert cause.search(line) is None, line


def test_negative_and_zero_change_formatting():
    down = pc.PriceChange(-1.234, 100, 98.766, NOW - timedelta(hours=3), NOW)
    assert "−1,23%" in pc.coincidence_line(down)
    flat = pc.PriceChange(0.0, 100, 100, NOW - timedelta(hours=3), NOW)
    assert "на 0%" in pc.coincidence_line(flat)


# --- Дополнение: разбор момента и наивное время публикации ----------------------


def test_moment_parsing_cases():
    assert pc._moment(None) is None
    assert pc._moment(5) is None
    assert pc._moment("") is None
    assert pc._moment("не дата") is None
    aware = pc._moment("2026-10-16T10:00:00+03:00")
    assert aware.utcoffset() == timedelta(hours=3)
    naive = pc._moment("2026-10-16T10:00:00")
    assert naive.tzinfo is not None and naive.utcoffset() == timedelta(hours=3)  # без пояса — МСК


def test_prices_skip_candles_with_missing_or_bad_values():
    assert pc._prices({"begin": "2026-10-16T10:00:00+03:00", "open": 1}) is None
    assert pc._prices({"begin": "2026-10-16T10:00:00+03:00", "open": "x", "close": 1}) is None
    assert pc._prices({"begin": "мусор", "open": 1, "close": 1}) is None
    assert pc._prices({"begin": "2026-10-16T10:00:00+03:00", "open": 1, "close": "2"})[2] == 2.0


def test_naive_publication_time_is_treated_as_utc():
    candles = [candle(h, 100.0 + h) for h in range(8, 17)]
    published_naive = datetime(2026, 10, 16, 8, 30)  # 08:30 UTC = 11:30 МСК
    published_utc = published_naive.replace(tzinfo=timezone.utc)
    assert pc.price_change_since(candles, published_naive, NOW) == pc.price_change_since(
        candles, published_utc, NOW
    )
    assert pc.price_change_since(candles, published_naive, NOW) is not None
