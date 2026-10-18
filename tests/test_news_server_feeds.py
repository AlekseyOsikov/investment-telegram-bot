"""Тесты разбора лент сервера новостей (mcp_servers/news/news_feeds.py) на сокращённых образцах
без сети. Компании и заголовки вымышленные; формат — как у публичных лент."""

import urllib.error
from datetime import timedelta, timezone

import news_feeds as nf
import pytest

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Лента</title>
<item><title>Альфатрейд объявила дивиденды</title>
<link>https://example.com/n/1?utm_source=rss</link>
<description>&lt;p&gt;Совет директоров рекомендовал выплату.&lt;/p&gt;</description>
<pubDate>Sat, 17 Oct 2026 09:30:00 +0300</pubDate></item>
<item><title></title><link>https://example.com/n/2</link></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Atom</title>
<entry><title>Бетагаз сменил руководителя</title>
<link href="https://example.org/a/5"/><updated>2026-10-17T06:00:00Z</updated>
<summary>Назначен новый председатель правления.</summary></entry></feed>"""

ISS = (
    '{"sitenews": {"columns": ["id", "tag", "title", "published_at", "modified_at"], '
    '"data": [[105206, "site", "Об изменении мер для ценной бумаги ZZZZ", '
    '"2026-10-16 22:30:00", "2026-10-18 11:47:07"], [1, "site", "", "2026-10-16 10:00:00", ""]]}}'
)


def test_rss_item_parsed_and_cleaned():
    items = nf.parse_feed(RSS, "example.com")
    assert len(items) == 1  # запись без заголовка пропущена
    item = items[0]
    assert item.title == "Альфатрейд объявила дивиденды"
    assert item.summary == "Совет директоров рекомендовал выплату."
    assert item.published_at.utcoffset() == timedelta(hours=3)
    assert item.source == "example.com"


def test_atom_entry_parsed():
    item = nf.parse_feed(ATOM, "example.org")[0]
    assert item.link == "https://example.org/a/5"
    assert item.published_at.tzinfo is not None
    assert item.summary.startswith("Назначен")


def test_iss_sitenews_parsed_with_moscow_time():
    items = nf.parse_feed(ISS, "iss.moex.com")
    assert len(items) == 1
    assert items[0].link == "https://www.moex.com/n105206"
    assert items[0].published_at.utcoffset() == timedelta(hours=3)
    assert items[0].published_at.astimezone(timezone.utc).hour == 19


def test_bad_iss_and_bad_xml_raise_feed_error():
    with pytest.raises(nf.FeedError):
        nf.parse_feed('{"other": 1}', "x")
    with pytest.raises(nf.FeedError):
        nf.parse_feed("<rss><item>", "x")


def test_doctype_is_rejected():
    with pytest.raises(nf.FeedError):
        nf.parse_feed('<!DOCTYPE x [<!ENTITY a "b">]><rss></rss>', "x")


def test_one_failing_feed_does_not_break_others():
    bodies = {"https://a.example/rss": RSS, "https://b.example/iss.json": ISS}

    def fetch(url):
        if url == "https://down.example/rss":
            raise nf.FeedError("лента не отвечает (TimeoutError)")
        if url == "https://boom.example/rss":
            raise RuntimeError("неожиданно")
        return bodies[url]

    urls = [
        "https://a.example/rss",
        "https://down.example/rss",
        "https://boom.example/rss",
        "https://b.example/iss.json",
    ]
    items, reports = nf.collect_feeds(urls, fetch)
    assert len(items) == 2
    assert [r.ok for r in reports] == [True, False, False, True]
    assert "не отвечает" in reports[1].error
    assert reports[3].count == 1


def test_access_denied_is_reported_not_bypassed(monkeypatch):
    def deny(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(nf.urllib.request, "urlopen", deny)
    with pytest.raises(nf.FeedError, match="доступ закрыт"):
        nf.fetch_url("https://closed.example/rss")


def test_fetch_rejects_non_http_scheme():
    with pytest.raises(nf.FeedError):
        nf.fetch_url("file:///etc/passwd")


def test_source_name_strips_www():
    assert nf.source_name("https://www.example.com/rss") == "example.com"
