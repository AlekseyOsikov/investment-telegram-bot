"""Тесты чистых правил сервера новостей (mcp_servers/news/news_rules.py): идентификатор,
нормализация, привязка к тикерам. Названия компаний вымышленные."""

from datetime import datetime, timezone

import news_rules as nr
import pytest

TABLE = nr.parse_alias_table(
    {
        "ALFA": {"names": ["Альфатрейд", "Альфа Трейдинг"]},
        "BETA": {
            "names": ["Бетагаз"],
            "weak": [{"alias": "Бета", "context": ["акци", "дивиденд"]}],
            "excludes": ["Бетагаз нефть"],
        },
        "BETN": {"names": ["Бетагаз нефть"]},
    }
)


def test_same_link_in_two_feeds_gives_one_id():
    first = nr.news_id("https://Example.com/a/1?utm_source=rss#top", "Заголовок", "лента-1")
    second = nr.news_id("https://example.com/a/1", "Другой заголовок", "лента-2")
    assert first == second


def test_without_link_id_uses_title_and_source():
    one = nr.news_id(None, "  Заголовок   новости ", "лента")
    assert one == nr.news_id("", "заголовок новости", "ЛЕНТА")
    assert one != nr.news_id(None, "заголовок новости", "другая")


def test_non_http_link_is_not_a_link():
    assert nr.canonical_link("javascript:alert(1)") == ""
    assert nr.canonical_link("ftp://example.com/x") == ""


def test_normalize_text_strips_markup():
    assert (
        nr.normalize_text("<p>Дивиденды&nbsp;&amp; отчёт</p>\n  текст") == "Дивиденды & отчёт текст"
    )
    assert nr.normalize_text(None) == ""


def test_truncate_excerpt_limits_length():
    text = "слово " * 200
    cut = nr.truncate_excerpt(text, 50)
    assert len(cut) <= 50 and cut.endswith("…")
    assert nr.truncate_excerpt("коротко", 50) == "коротко"
    assert nr.truncate_excerpt("текст", 0) == ""


def test_name_match_with_inflection_names_the_match():
    found = nr.match_tickers("Совет директоров Альфатрейда утвердил отчёт", "", TABLE)
    assert [(m.ticker, m.kind) for m in found] == [("ALFA", nr.KIND_NAME)]
    assert found[0].matched.lower().startswith("альфатрейд")


def test_ticker_only_as_whole_uppercase_word():
    assert nr.match_tickers("Акции ALFA подорожали", "", TABLE)[0].kind == nr.KIND_TICKER
    assert nr.match_tickers("alfa-романтика", "", TABLE) == []
    assert nr.match_tickers("ALFABET растёт", "", TABLE) == []


def test_ambiguous_short_alias_needs_context():
    assert nr.match_tickers("Бета-блокаторы снова в моде", "", TABLE) == []
    found = nr.match_tickers("Бета объявила дивиденды", "", TABLE)
    assert [(m.ticker, m.kind) for m in found] == [("BETA", nr.KIND_NAME_CONTEXT)]


def test_exclude_prevents_wrong_company():
    found = nr.match_tickers("Бетагаз нефть заключила контракт", "", TABLE)
    assert [m.ticker for m in found] == ["BETN"]


def test_news_without_binding_gives_nothing():
    assert nr.match_tickers("Погода в Москве", "Обещают дождь", TABLE) == []


def test_extra_ticker_not_in_table_found_only_as_uppercase_word():
    found = nr.match_tickers("Бумага ZZZZ торгуется", "", TABLE, extra_tickers=("ZZZZ",))
    assert [(m.ticker, m.kind) for m in found] == [("ZZZZ", nr.KIND_TICKER)]
    assert nr.match_tickers("zzzz", "", TABLE, extra_tickers=("ZZZZ",)) == []


def test_matching_looks_in_body_too():
    found = nr.match_tickers("Итоги дня", "Среди прочего — Альфа Трейдинг", TABLE)
    assert [m.ticker for m in found] == ["ALFA"]


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"alfa": {"names": ["X"]}},
        {"ALFA": {"names": []}},
        {"ALFA": {"weak": [{"alias": "Аль", "context": []}]}},
        {"ALFA": {"names": "не список"}},
    ],
)
def test_alias_table_schema_violations(data):
    with pytest.raises(ValueError):
        nr.parse_alias_table(data)


def test_msk_date_uses_moscow_day():
    late_utc = datetime(2026, 10, 18, 22, 30, tzinfo=timezone.utc)  # 01:30 МСК следующих суток
    assert nr.msk_date(late_utc) == "2026-10-19"
