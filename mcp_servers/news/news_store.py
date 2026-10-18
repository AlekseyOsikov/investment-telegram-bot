"""Хранилище сервера новостей на SQLite (спека news-server, требование «Хранение только публичных
данных»): новости, привязка к тикерам, результат разбора, включённость дайджеста чата, отметки
доставки и счётчик сообщений за московские сутки.

Хранится только публичное (идентификатор, ссылка, заголовок, короткий фрагмент, время, привязка,
разбор) и по чатам — chat_id, признак включённости, отметки доставки. Сообщений пользователей,
сумм и портфеля нет. Время всегда передаётся аргументом (`now`), поэтому всё проверяется тестами на
временной БД (`tests/test_news_server_store.py`). Не часть бота: `src/` этот модуль не импортирует.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import news_rules as nr

IMPORTANCE_RANK = {"high": 0, "medium": 1, "low": 2}

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    link TEXT NOT NULL,
    title TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    published_at TEXT NOT NULL,
    collected_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_tickers (
    news_id TEXT NOT NULL REFERENCES news(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    matched TEXT NOT NULL,
    kind TEXT NOT NULL,
    PRIMARY KEY (news_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_news_tickers_ticker ON news_tickers(ticker);
CREATE TABLE IF NOT EXISTS analyses (
    news_id TEXT PRIMARY KEY REFERENCES news(id) ON DELETE CASCADE,
    available INTEGER NOT NULL,
    event_type TEXT,
    importance TEXT,
    summary TEXT,
    mechanism TEXT,
    analyzed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    chat_id INTEGER PRIMARY KEY,
    enabled_at TEXT NOT NULL,
    last_digest_at TEXT
);
CREATE TABLE IF NOT EXISTS deliveries (
    chat_id INTEGER NOT NULL,
    news_id TEXT NOT NULL REFERENCES news(id) ON DELETE CASCADE,
    delivered_at TEXT NOT NULL,
    PRIMARY KEY (chat_id, news_id)
);
CREATE TABLE IF NOT EXISTS daily_counts (
    chat_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    messages INTEGER NOT NULL,
    PRIMARY KEY (chat_id, day)
);
"""


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class NewsRecord:
    """Новость к сохранению: уже с привязкой, найденной кодом."""

    id: str
    source: str
    link: str
    title: str
    excerpt: str
    published_at: datetime | None
    matches: tuple[nr.TickerMatch, ...]


class NewsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # --- новости ---------------------------------------------------------------------

    def add_news(self, records: Iterable[NewsRecord], now: datetime) -> dict[str, int]:
        """Сохраняет новости с привязкой; повторная новость (тот же id) не дублируется, но
        может получить новые привязки. Новость без привязки не сохраняется вовсе: она нужна
        только в наборах по тикерам. Возвращает {"added": N, "bound": M}."""
        added = 0
        seen = 0
        with closing(self._connect()) as conn, conn:
            for record in records:
                seen += 1 if record.matches else 0
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO news(id, source, link, title, excerpt, published_at,"
                    " collected_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.source,
                        record.link,
                        record.title,
                        record.excerpt,
                        _iso(record.published_at or now),
                        _iso(now),
                    ),
                )
                added += cursor.rowcount
                for match in record.matches:
                    conn.execute(
                        "INSERT OR IGNORE INTO news_tickers(news_id, ticker, matched, kind)"
                        " VALUES (?, ?, ?, ?)",
                        (record.id, match.ticker, match.matched, match.kind),
                    )
        return {"added": added, "bound": seen}

    def _tickers_of(self, conn: sqlite3.Connection, news_id: str) -> list[dict]:
        rows = conn.execute(
            "SELECT ticker, matched, kind FROM news_tickers WHERE news_id = ? ORDER BY ticker",
            (news_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending(
        self, tickers: list[str], now: datetime, max_age_hours: int, limit: int
    ) -> list[dict]:
        """Новости по тикерам без результата разбора, свежие первыми, не старше окна."""
        if not tickers:
            return []
        marks = ",".join("?" * len(tickers))
        since = _iso(now - timedelta(hours=max_age_hours))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT n.id, n.source, n.link, n.title, n.excerpt, n.published_at"
                f" FROM news n JOIN news_tickers t ON t.news_id = n.id"
                f" LEFT JOIN analyses a ON a.news_id = n.id"
                f" WHERE a.news_id IS NULL AND t.ticker IN ({marks}) AND n.published_at >= ?"
                " ORDER BY n.published_at DESC LIMIT ?",
                (*tickers, since, limit),
            ).fetchall()
            return [{**dict(row), "tickers": self._tickers_of(conn, row["id"])} for row in rows]

    def save_analysis(self, news_id: str, analysis: dict | None, now: datetime) -> bool:
        """Сохраняет разбор (`None` — «разбор недоступен»). False, если новости нет."""
        with closing(self._connect()) as conn, conn:
            if conn.execute("SELECT 1 FROM news WHERE id = ?", (news_id,)).fetchone() is None:
                return False
            conn.execute(
                "INSERT OR REPLACE INTO analyses(news_id, available, event_type, importance,"
                " summary, mechanism, analyzed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    news_id,
                    0 if analysis is None else 1,
                    (analysis or {}).get("event_type"),
                    (analysis or {}).get("importance"),
                    (analysis or {}).get("summary"),
                    (analysis or {}).get("mechanism"),
                    _iso(now),
                ),
            )
        return True

    def ready(
        self,
        chat_id: int,
        tickers: list[str],
        now: datetime,
        max_age_hours: int,
        limit: int,
        urgent_only: bool = False,
    ) -> list[dict]:
        """Разобранные (в том числе «без разбора») и ещё не отправленные в чат новости по
        тикерам: сначала важные, затем свежие. Срочные — только с важностью high."""
        if not tickers:
            return []
        marks = ",".join("?" * len(tickers))
        since = _iso(now - timedelta(hours=max_age_hours))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT n.id, n.source, n.link, n.title, n.published_at,"
                " a.available, a.event_type, a.importance, a.summary, a.mechanism"
                " FROM news n JOIN news_tickers t ON t.news_id = n.id"
                " JOIN analyses a ON a.news_id = n.id"
                f" WHERE t.ticker IN ({marks}) AND n.published_at >= ?"
                " AND NOT EXISTS (SELECT 1 FROM deliveries d"
                "  WHERE d.chat_id = ? AND d.news_id = n.id)",
                (*tickers, since, chat_id),
            ).fetchall()
            items = []
            for row in rows:
                if urgent_only and not (row["available"] and row["importance"] == "high"):
                    continue
                analysis = (
                    {
                        "event_type": row["event_type"],
                        "importance": row["importance"],
                        "summary": row["summary"],
                        "mechanism": row["mechanism"],
                    }
                    if row["available"]
                    else None
                )
                mine = [t for t in self._tickers_of(conn, row["id"]) if t["ticker"] in tickers]
                items.append(
                    {
                        "id": row["id"],
                        "source": row["source"],
                        "link": row["link"],
                        "title": row["title"],
                        "published_at": row["published_at"],
                        "tickers": mine,
                        "analysis": analysis,
                    }
                )
        items.sort(key=lambda item: item["published_at"], reverse=True)
        items.sort(
            key=lambda item: IMPORTANCE_RANK.get((item["analysis"] or {}).get("importance"), 3)
        )
        return items[:limit]

    # --- чаты, доставка, лимит ---------------------------------------------------------

    def enable_chat(self, chat_id: int, now: datetime) -> dict:
        """Включает дайджест чата; повторное включение срок не сдвигает."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO chats(chat_id, enabled_at) VALUES (?, ?)",
                (chat_id, _iso(now)),
            )
        return self.chat_state(chat_id, now)

    def disable_chat(self, chat_id: int) -> bool:
        """Выключает дайджест и удаляет данные чата (отметки доставки и счётчики).
        True, если дайджест был включён."""
        with closing(self._connect()) as conn, conn:
            existed = conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,)).rowcount
            conn.execute("DELETE FROM deliveries WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM daily_counts WHERE chat_id = ?", (chat_id,))
        return bool(existed)

    def enabled_chats(self) -> list[int]:
        with closing(self._connect()) as conn:
            return [
                r["chat_id"] for r in conn.execute("SELECT chat_id FROM chats ORDER BY chat_id")
            ]

    def chat_state(self, chat_id: int, now: datetime) -> dict:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM chats WHERE chat_id = ?", (chat_id,)).fetchone()
            today = nr.msk_date(now)
            count = conn.execute(
                "SELECT messages FROM daily_counts WHERE chat_id = ? AND day = ?",
                (chat_id, today),
            ).fetchone()
        return {
            "enabled": row is not None,
            "enabled_at": row["enabled_at"] if row else None,
            "last_digest_at": row["last_digest_at"] if row else None,
            "messages_today": count["messages"] if count else 0,
            "day": today,
        }

    def ack(self, chat_id: int, news_ids: list[str], now: datetime, kind: str = "digest") -> dict:
        """Подтверждает доставку ОДНОГО сообщения: отметки по новостям, +1 к суточному
        счётчику, для дайджеста — момент последнего дайджеста. Вызывается ПОСЛЕ отправки."""
        with closing(self._connect()) as conn, conn:
            if conn.execute("SELECT 1 FROM chats WHERE chat_id = ?", (chat_id,)).fetchone() is None:
                return {"acked": 0}
            for news_id in news_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO deliveries(chat_id, news_id, delivered_at)"
                    " SELECT ?, id, ? FROM news WHERE id = ?",
                    (chat_id, _iso(now), news_id),
                )
            conn.execute(
                "INSERT INTO daily_counts(chat_id, day, messages) VALUES (?, ?, 1)"
                " ON CONFLICT(chat_id, day) DO UPDATE SET messages = messages + 1",
                (chat_id, nr.msk_date(now)),
            )
            if kind == "digest":
                conn.execute(
                    "UPDATE chats SET last_digest_at = ? WHERE chat_id = ?", (_iso(now), chat_id)
                )
        return {"acked": len(news_ids)}

    # --- срок хранения -----------------------------------------------------------------

    def purge(self, now: datetime, retention_days: int) -> int:
        """Удаляет новости старше срока хранения вместе с привязкой, разбором и отметками
        доставки, а также устаревшие суточные счётчики. Возвращает число удалённых новостей."""
        border = _iso(now - timedelta(days=retention_days))
        with closing(self._connect()) as conn, conn:
            removed = conn.execute("DELETE FROM news WHERE collected_at < ?", (border,)).rowcount
            conn.execute(
                "DELETE FROM daily_counts WHERE day < ?",
                (nr.msk_date(now - timedelta(days=retention_days)),),
            )
        return removed
