"""Тесты хранилища сервера новостей (mcp_servers/news/news_store.py) на временной БД."""

from datetime import datetime, timedelta, timezone

import news_rules as nr
import news_store as ns
import pytest

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return ns.NewsStore(tmp_path / "sub" / "news.sqlite3")


def record(
    link="https://example.com/1",
    title="Альфатрейд объявила дивиденды",
    tickers=("ALFA",),
    published=NOW - timedelta(hours=1),
):
    matches = tuple(nr.TickerMatch(t, t, nr.KIND_TICKER) for t in tickers)
    return ns.NewsRecord(
        nr.news_id(link, title, "example.com"),
        "example.com",
        link,
        title,
        "фрагмент",
        published,
        matches,
    )


ANALYSIS = {"event_type": "dividends", "importance": "high", "summary": "С.", "mechanism": "М."}


def test_same_news_stored_once_and_bindings_merged(store):
    assert store.add_news([record()], NOW) == {"added": 1, "bound": 1}
    assert store.add_news([record(tickers=("ALFA", "BETA"))], NOW)["added"] == 0
    item = store.pending(["BETA"], NOW, 48, 10)[0]
    assert {t["ticker"] for t in item["tickers"]} == {"ALFA", "BETA"}


def test_news_without_binding_stored_but_never_returned(store):
    assert store.add_news([record(tickers=())], NOW) == {"added": 1, "bound": 0}
    assert store.pending(["ALFA"], NOW, 48, 10) == []
    assert store.add_news([record(tickers=("ALFA",))], NOW)["added"] == 0  # привязка добавилась
    assert len(store.pending(["ALFA"], NOW, 48, 10)) == 1


def test_pending_excludes_analyzed_and_old(store):
    old = record("https://example.com/old", "Старое", published=NOW - timedelta(hours=100))
    store.add_news([record(), old], NOW)
    assert [i["link"] for i in store.pending(["ALFA"], NOW, 48, 10)] == ["https://example.com/1"]
    store.save_analysis(record().id, ANALYSIS, NOW)
    assert store.pending(["ALFA"], NOW, 48, 10) == []


def test_save_analysis_unknown_news(store):
    assert store.save_analysis("нет-такой", ANALYSIS, NOW) is False


def test_ready_returns_unconfirmed_until_ack(store):
    store.add_news([record()], NOW)
    store.enable_chat(7, NOW)
    assert store.ready(7, ["ALFA"], NOW, 48, 10) == []  # ещё нет разбора
    store.save_analysis(record().id, ANALYSIS, NOW)
    first = store.ready(7, ["ALFA"], NOW, 48, 10)
    assert len(first) == 1 and first[0]["analysis"]["importance"] == "high"
    assert store.ready(7, ["ALFA"], NOW, 48, 10) == first  # без подтверждения — выдаётся снова
    store.ack(7, [first[0]["id"]], NOW)
    assert store.ready(7, ["ALFA"], NOW, 48, 10) == []
    assert store.ready(8, ["ALFA"], NOW, 48, 10) != []  # другой чат не затронут


def test_unavailable_analysis_is_ready_without_analysis(store):
    store.add_news([record()], NOW)
    store.save_analysis(record().id, None, NOW)
    item = store.ready(1, ["ALFA"], NOW, 48, 10)[0]
    assert item["analysis"] is None


def test_ready_orders_by_importance_then_freshness_and_urgent_filter(store):
    a = record("https://e.com/a", "А", published=NOW - timedelta(hours=3))
    b = record("https://e.com/b", "Б", published=NOW - timedelta(hours=1))
    c = record("https://e.com/c", "В", published=NOW - timedelta(hours=2))
    store.add_news([a, b, c], NOW)
    store.save_analysis(a.id, {**ANALYSIS, "importance": "high"}, NOW)
    store.save_analysis(b.id, {**ANALYSIS, "importance": "low"}, NOW)
    store.save_analysis(c.id, {**ANALYSIS, "importance": "low"}, NOW)
    order = [i["link"] for i in store.ready(1, ["ALFA"], NOW, 48, 10)]
    assert order == ["https://e.com/a", "https://e.com/b", "https://e.com/c"]
    urgent = store.ready(1, ["ALFA"], NOW, 48, 10, urgent_only=True)
    assert [i["link"] for i in urgent] == ["https://e.com/a"]
    assert len(store.ready(1, ["ALFA"], NOW, 48, 2)) == 2


def test_ready_shows_only_requested_tickers_of_the_news(store):
    store.add_news([record(tickers=("ALFA", "BETA"))], NOW)
    store.save_analysis(record().id, ANALYSIS, NOW)
    item = store.ready(1, ["BETA"], NOW, 48, 10)[0]
    assert [t["ticker"] for t in item["tickers"]] == ["BETA"]


def test_daily_counter_by_moscow_day_and_last_digest(store):
    store.add_news([record()], NOW)
    store.enable_chat(5, NOW)
    store.ack(5, [record().id], NOW, kind="urgent")
    state = store.chat_state(5, NOW)
    assert state["messages_today"] == 1 and state["last_digest_at"] is None
    store.ack(5, [], NOW, kind="digest")
    state = store.chat_state(5, NOW)
    assert state["messages_today"] == 2 and state["last_digest_at"] is not None
    next_day = NOW + timedelta(days=1)
    assert store.chat_state(5, next_day)["messages_today"] == 0


def test_ack_for_disabled_chat_is_ignored(store):
    store.add_news([record()], NOW)
    assert store.ack(99, [record().id], NOW) == {"acked": 0}
    assert store.chat_state(99, NOW)["messages_today"] == 0


def test_enable_is_idempotent_and_disable_removes_chat_data(store):
    store.add_news([record()], NOW)
    store.save_analysis(record().id, ANALYSIS, NOW)
    first = store.enable_chat(3, NOW)
    again = store.enable_chat(3, NOW + timedelta(hours=5))
    assert first["enabled_at"] == again["enabled_at"]
    store.ack(3, [record().id], NOW)
    assert store.disable_chat(3) is True
    assert store.disable_chat(3) is False  # повторная остановка — не ошибка
    state = store.chat_state(3, NOW)
    assert state["enabled"] is False and state["messages_today"] == 0
    store.enable_chat(3, NOW)
    assert len(store.ready(3, ["ALFA"], NOW, 48, 10)) == 1  # отметки удалены


def test_disable_keeps_other_chats_and_public_news(store):
    store.add_news([record()], NOW)
    store.save_analysis(record().id, ANALYSIS, NOW)
    store.enable_chat(1, NOW)
    store.enable_chat(2, NOW)
    store.ack(2, [record().id], NOW)
    store.disable_chat(1)
    assert store.enabled_chats() == [2]
    assert store.ready(2, ["ALFA"], NOW, 48, 10) == []
    assert store.ready(1, ["ALFA"], NOW, 48, 10) != []


def test_purge_removes_old_news_with_everything_attached(store):
    old_time = NOW - timedelta(days=10)
    store.add_news([record("https://e.com/old", "Старое", published=old_time)], old_time)
    store.add_news([record()], NOW)
    store.enable_chat(1, old_time)
    store.save_analysis(
        nr.news_id("https://e.com/old", "Старое", "example.com"), ANALYSIS, old_time
    )
    store.ack(1, [nr.news_id("https://e.com/old", "Старое", "example.com")], old_time)
    assert store.purge(NOW, 7) == 1
    assert [i["link"] for i in store.pending(["ALFA"], NOW, 24 * 30, 10)] == [
        "https://example.com/1"
    ]
    assert store.chat_state(1, old_time)["messages_today"] == 0  # счётчик давних суток убран
