"""Тесты обработчиков сервера новостей (mcp_servers/news/news_server.py): `NewsService` на
временной БД и фейковой загрузке лент, таблица алиасов, набор инструментов."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import news_feeds as nf
import news_rules as nr
import news_server as srv
import news_store as ns
import pytest

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
ALIASES = Path(__file__).resolve().parent.parent / "mcp_servers" / "news" / "data" / "aliases.json"
TABLE = nr.parse_alias_table({"ALFA": {"names": ["Альфатрейд"]}, "BETA": {"names": ["Бетагаз"]}})

RSS_ONE = """<rss><channel>
<item><title>Альфатрейд объявила дивиденды</title><link>https://a.example/1?utm_x=1</link>
<description>Текст</description><pubDate>Sun, 18 Oct 2026 14:00:00 +0300</pubDate></item>
<item><title>Погода</title><link>https://a.example/2</link></item></channel></rss>"""
RSS_TWO = """<rss><channel>
<item><title>Альфатрейд объявила дивиденды</title><link>https://a.example/1</link>
<pubDate>Sun, 18 Oct 2026 14:00:00 +0300</pubDate></item>
<item><title>ZZZZ: меры биржи</title><link>https://b.example/9</link>
<pubDate>Sun, 18 Oct 2026 13:00:00 +0300</pubDate></item></channel></rss>"""


def make_service(tmp_path, bodies=None, retention=7):
    bodies = bodies or {"https://a.example/rss": RSS_ONE, "https://b.example/rss": RSS_TWO}

    def fetch(url):
        body = bodies[url]
        if isinstance(body, Exception):
            raise body
        return body

    return srv.NewsService(
        ns.NewsStore(tmp_path / "news.sqlite3"), list(bodies), TABLE, retention, fetch, lambda: NOW
    )


def test_collect_dedups_across_feeds_and_binds(tmp_path):
    service = make_service(tmp_path)
    result = service.collect(["ZZZZ"])
    assert result["bound"] == 2 and result["added"] == 3  # «Погода» хранится без привязки
    items = service.pending(["ALFA", "ZZZZ"])["items"]
    assert len(items) == 2
    assert service.collect(["ZZZZ"])["added"] == 0  # повторный сбор ничего не дублирует


def test_unavailable_feed_is_reported_and_others_collected(tmp_path):
    service = make_service(
        tmp_path,
        {"https://a.example/rss": RSS_ONE, "https://down.example/rss": nf.FeedError("не отвечает")},
    )
    result = service.collect()
    assert result["added"] >= 1
    assert [f["url"] for f in result["feeds_unavailable"]] == ["https://down.example/rss"]


def test_full_cycle_collect_analyze_ready_ack(tmp_path):
    service = make_service(tmp_path)
    service.collect(["ZZZZ"])
    service.enable(10)
    news = service.pending(["ALFA"])["items"][0]
    service.save_analysis(news["id"], True, "dividends", "high", "Пересказ.", "Механизм.")
    ready = service.ready(10, ["ALFA"])["items"]
    assert ready[0]["analysis"]["event_type"] == "dividends"
    service.ack(10, [ready[0]["id"]], "urgent")
    assert service.ready(10, ["ALFA"])["items"] == []
    assert service.state(10)["messages_today"] == 1


def test_validation_errors_are_russian_news_errors(tmp_path):
    service = make_service(tmp_path)
    with pytest.raises(srv.NewsError, match="тикер"):
        service.pending(["bad ticker!"])
    with pytest.raises(srv.NewsError):
        service.pending([])
    with pytest.raises(srv.NewsError, match="limit"):
        service.pending(["ALFA"], limit=0)
    with pytest.raises(srv.NewsError, match="chat_id"):
        service.state("1")  # type: ignore[arg-type]
    with pytest.raises(srv.NewsError, match="importance"):
        service.save_analysis("x", True, "dividends", "extreme", "С.", "М.")
    with pytest.raises(srv.NewsError, match="не найдена"):
        service.save_analysis("x", True, "dividends", "high", "С.", "М.")
    with pytest.raises(srv.NewsError, match="kind"):
        service.ack(1, [], "weird")


def test_unavailable_analysis_saved(tmp_path):
    service = make_service(tmp_path)
    service.collect()
    news = service.pending(["ALFA"])["items"][0]
    assert service.save_analysis(news["id"], False)["available"] is False
    assert service.ready(1, ["ALFA"])["items"][0]["analysis"] is None


def test_disable_reports_and_is_idempotent(tmp_path):
    service = make_service(tmp_path)
    service.enable(4)
    assert service.disable(4) == {"was_enabled": True}
    assert service.disable(4) == {"was_enabled": False}
    assert service.chats() == {"chat_ids": []}


def test_coverage_splits_known_and_unknown_tickers(tmp_path):
    service = make_service(tmp_path)
    assert service.coverage(["alfa", "zzzz"]) == {"covered": ["ALFA"], "uncovered": ["ZZZZ"]}


def test_retention_purges_old_records(tmp_path):
    service = make_service(tmp_path, retention=1)
    old = NOW - timedelta(days=3)
    service.store.add_news(
        [
            ns.NewsRecord(
                "old",
                "s",
                "https://x.example/o",
                "Старое",
                "",
                old,
                (nr.TickerMatch("ALFA", "ALFA", nr.KIND_TICKER),),
            )
        ],
        old,
    )
    assert service.collect()["purged"] == 1


def test_shipped_alias_table_is_valid_and_has_no_user_data():
    table = nr.parse_alias_table(json.loads(ALIASES.read_text(encoding="utf-8")))
    assert {"SBER", "GAZP", "SIBN"} <= set(table)
    gazprom = nr.match_tickers("Газпром нефть увеличила добычу", "", table)
    assert [m.ticker for m in gazprom] == ["SIBN"]
    assert (
        nr.match_tickers("Сбер обновил приложение", "", table) == []
    )  # слабый алиас без контекста


def test_tool_set_and_no_read_only_marks(tmp_path):
    server = srv.build_server(make_service(tmp_path))
    tools = asyncio.run(server.list_tools())
    tools = tools.tools if hasattr(tools, "tools") else tools
    assert {t.name for t in tools} == {
        "news_collect",
        "news_pending",
        "news_save_analysis",
        "news_ready",
        "news_ack",
        "news_enable",
        "news_disable",
        "news_state",
        "news_chats",
        "news_coverage",
    }
    # ни один инструмент не должен выглядеть «безопасным для модели»
    assert all(not (t.annotations and t.annotations.read_only_hint) for t in tools)


def test_args_require_absolute_db_and_service_builds(tmp_path):
    args = srv.parse_args(["--db", str(tmp_path / "n.sqlite3"), "--feeds", "https://a.example/rss"])
    assert srv.service_from_args(args).feeds == ["https://a.example/rss"]
    with pytest.raises(SystemExit):
        srv.service_from_args(srv.parse_args(["--db", "relative.sqlite3"]))
