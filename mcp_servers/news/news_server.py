"""MCP-сервер новостей для дайджеста (спеки news-server и news-digest; design.md изменения
add-news-digest, решения 2–4): забирает публичные ленты, убирает повторы, привязывает новости к
тикерам по таблице алиасов и хранит публичные записи в SQLite.

«Пассивный» сервер, как mcp-moex в режиме расписания: время не измеряет, сообщений не отправляет,
модель не вызывает. Все границы задаёт ОПЕРАТОР аргументами запуска (путь БД, адреса лент, срок
хранения) — в схемах инструментов их нет. Инструменты принимают chat_id и тикеры от бота и
изменяют хранилище, поэтому НИ ОДИН слой бота не отдаёт их модели. Сервер не читает опросы цен
других проектов: тикеры чата приходят аргументом. Не часть бота: `src/` этот модуль не
импортирует.

Запуск: `python mcp_servers/news/news_server.py --db <абсолютный путь> --feeds <url> [<url> ...]`
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import news_feeds as nf
import news_rules as nr
import news_store as ns
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

# Ни один инструмент не помечается read-only: «read-only» в проекте значит «можно отдавать
# модели», а здесь нельзя ничего (ни выдачу, ни состояние).
STATE_TOOL = ToolAnnotations(read_only_hint=False, destructive_hint=False)
DELETES_CHAT_DATA = ToolAnnotations(read_only_hint=False, destructive_hint=True)

DEFAULT_ALIASES = Path(__file__).resolve().parent / "data" / "aliases.json"
DEFAULT_RETENTION_DAYS = 7
MAX_TICKERS = 50
MAX_LIMIT = 100
DEFAULT_MAX_AGE_HOURS = 48
_TICKER_RE = re.compile(r"[A-Z0-9]{2,12}")
_ANALYSIS_FIELDS = ("event_type", "importance", "summary", "mechanism")
_MAX_FIELD = 600


class NewsError(Exception):
    """Ошибка обращения к серверу: `str(error)` — понятный русский текст."""


def _tickers(value: list[str]) -> list[str]:
    if not isinstance(value, list) or not 0 < len(value) <= MAX_TICKERS:
        raise NewsError(f"Нужен список тикеров: от 1 до {MAX_TICKERS} штук.")
    result: list[str] = []
    for raw in value:
        ticker = str(raw).strip().upper()
        if not _TICKER_RE.fullmatch(ticker):
            raise NewsError(f"Некорректный тикер {raw!r}: нужны латинские буквы и цифры.")
        if ticker not in result:
            result.append(ticker)
    return result


def _chat_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NewsError("chat_id должен быть целым числом.")
    return value


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_LIMIT:
        raise NewsError(f"limit должен быть целым числом от 1 до {MAX_LIMIT}.")
    return value


class NewsService:
    """Обработчики инструментов без MCP: тестируются без процесса и сети. Каждый возвращает
    словарь либо поднимает `NewsError`."""

    def __init__(
        self,
        store: ns.NewsStore,
        feeds: list[str],
        table: dict[str, nr.AliasEntry],
        retention_days: int = DEFAULT_RETENTION_DAYS,
        fetch: nf.Fetch | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.feeds = list(feeds)
        self.table = table
        self.retention_days = retention_days
        self._fetch = fetch or nf.fetch_url
        self._now = clock or (lambda: datetime.now(timezone.utc))

    def collect(self, tickers: list[str] | None = None) -> dict[str, Any]:
        extra = tuple(_tickers(tickers)) if tickers else ()
        now = self._now()
        items, reports = nf.collect_feeds(self.feeds, self._fetch)
        records: dict[str, ns.NewsRecord] = {}
        for item in items:
            matches = nr.match_tickers(item.title, item.summary, self.table, extra)
            record = ns.NewsRecord(
                nr.news_id(item.link, item.title, item.source),
                item.source,
                nr.canonical_link(item.link) or item.link,
                item.title,
                nr.truncate_excerpt(item.summary),
                item.published_at,
                tuple(matches),
            )
            known = records.get(record.id)
            if known is not None:  # та же новость из другой ленты: объединяем привязки
                merged = {m.ticker: m for m in (*known.matches, *record.matches)}
                record = ns.NewsRecord(**{**known.__dict__, "matches": tuple(merged.values())})
            records[record.id] = record
        stored = self.store.add_news(records.values(), now)
        purged = self.store.purge(now, self.retention_days)
        unavailable = [r.as_dict() for r in reports if not r.ok]
        return {
            **stored,
            "purged": purged,
            "feeds_total": len(reports),
            "feeds_unavailable": unavailable,
        }

    def pending(
        self, tickers: list[str], limit: int = 20, max_age_hours: int = DEFAULT_MAX_AGE_HOURS
    ) -> dict[str, Any]:
        items = self.store.pending(_tickers(tickers), self._now(), max_age_hours, _limit(limit))
        return {"items": items}

    def save_analysis(
        self,
        news_id: str,
        available: bool = True,
        event_type: str = "",
        importance: str = "",
        summary: str = "",
        mechanism: str = "",
    ) -> dict[str, Any]:
        analysis = None
        if available:
            analysis = {
                "event_type": event_type,
                "importance": importance,
                "summary": summary,
                "mechanism": mechanism,
            }
            for name in _ANALYSIS_FIELDS:
                if not analysis[name] or len(analysis[name]) > _MAX_FIELD:
                    raise NewsError(f"Поле {name} разбора пустое или слишком длинное.")
            if importance not in ns.IMPORTANCE_RANK:
                raise NewsError("importance должна быть одной из: high, medium, low.")
        if not self.store.save_analysis(news_id, analysis, self._now()):
            raise NewsError(
                "Новость с таким идентификатором не найдена (возможно, удалена по сроку)."
            )
        return {"saved": True, "available": available}

    def ready(
        self,
        chat_id: int,
        tickers: list[str],
        limit: int = 20,
        urgent_only: bool = False,
        max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
    ) -> dict[str, Any]:
        items = self.store.ready(
            _chat_id(chat_id),
            _tickers(tickers),
            self._now(),
            max_age_hours,
            _limit(limit),
            urgent_only,
        )
        return {"items": items}

    def ack(self, chat_id: int, news_ids: list[str], kind: str = "digest") -> dict[str, Any]:
        if kind not in ("digest", "urgent"):
            raise NewsError("kind должен быть digest или urgent.")
        return self.store.ack(_chat_id(chat_id), list(news_ids), self._now(), kind)

    def enable(self, chat_id: int) -> dict[str, Any]:
        return self.store.enable_chat(_chat_id(chat_id), self._now())

    def disable(self, chat_id: int) -> dict[str, Any]:
        return {"was_enabled": self.store.disable_chat(_chat_id(chat_id))}

    def state(self, chat_id: int) -> dict[str, Any]:
        return self.store.chat_state(_chat_id(chat_id), self._now())

    def chats(self) -> dict[str, Any]:
        return {"chat_ids": self.store.enabled_chats()}

    def coverage(self, tickers: list[str]) -> dict[str, Any]:
        """Какие бумаги есть в таблице алиасов (ищутся и по названию), а какие — только по
        тикеру в тексте («покрытие ограничено»)."""
        names = _tickers(tickers)
        return {
            "covered": [t for t in names if t in self.table],
            "uncovered": [t for t in names if t not in self.table],
        }


def build_server(service: NewsService) -> MCPServer:
    """Регистрирует инструменты. `NewsError` превращается в `ToolError` (текст доходит до бота)."""
    server = MCPServer("news")

    def guarded(call):
        try:
            return call()
        except NewsError as exc:
            raise ToolError(str(exc)) from None

    @server.tool(annotations=STATE_TOOL)
    def news_collect(tickers: list[str] | None = None) -> dict[str, Any]:
        """Забрать настроенные ленты, убрать повторы, привязать новости к тикерам по таблице
        алиасов и сохранить. tickers — наблюдаемые бумаги вне таблицы (ищутся по тикеру в тексте).
        Результат называет недоступные ленты; устаревшие записи удаляются."""
        return guarded(lambda: service.collect(tickers))

    @server.tool(annotations=STATE_TOOL)
    def news_pending(
        tickers: list[str], limit: int = 20, max_age_hours: int = DEFAULT_MAX_AGE_HOURS
    ) -> dict[str, Any]:
        """Новости по тикерам, у которых ещё нет результата разбора (свежие первыми)."""
        return guarded(lambda: service.pending(tickers, limit, max_age_hours))

    @server.tool(annotations=STATE_TOOL)
    def news_save_analysis(
        news_id: str,
        available: bool = True,
        event_type: str = "",
        importance: str = "",
        summary: str = "",
        mechanism: str = "",
    ) -> dict[str, Any]:
        """Сохранить результат разбора новости (общий для всех чатов). available=false —
        «разбор недоступен»: новость будет показана без разбора."""
        return guarded(
            lambda: service.save_analysis(
                news_id, available, event_type, importance, summary, mechanism
            )
        )

    @server.tool(annotations=STATE_TOOL)
    def news_ready(
        chat_id: int,
        tickers: list[str],
        limit: int = 20,
        urgent_only: bool = False,
        max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
    ) -> dict[str, Any]:
        """Разобранные и ещё не подтверждённые для чата новости по тикерам: сначала важные.
        urgent_only — только с важностью high. Без news_ack выдаются повторно."""
        return guarded(lambda: service.ready(chat_id, tickers, limit, urgent_only, max_age_hours))

    @server.tool(annotations=STATE_TOOL)
    def news_ack(chat_id: int, news_ids: list[str], kind: str = "digest") -> dict[str, Any]:
        """Подтвердить доставку ОДНОГО сообщения чату (после успешной отправки): отметки по
        новостям и +1 к суточному счётчику. kind: digest | urgent."""
        return guarded(lambda: service.ack(chat_id, news_ids, kind))

    @server.tool(annotations=STATE_TOOL)
    def news_enable(chat_id: int) -> dict[str, Any]:
        """Включить дайджест чата."""
        return guarded(lambda: service.enable(chat_id))

    @server.tool(annotations=DELETES_CHAT_DATA)
    def news_disable(chat_id: int) -> dict[str, Any]:
        """Выключить дайджест чата и удалить отметки доставки и счётчики этого чата."""
        return guarded(lambda: service.disable(chat_id))

    @server.tool(annotations=STATE_TOOL)
    def news_state(chat_id: int) -> dict[str, Any]:
        """Состояние дайджеста чата: включён ли, момент последнего дайджеста, сообщений за
        московские сутки."""
        return guarded(lambda: service.state(chat_id))

    @server.tool(annotations=STATE_TOOL)
    def news_chats() -> dict[str, Any]:
        """Чаты с включённым дайджестом."""
        return guarded(service.chats)

    @server.tool(annotations=STATE_TOOL)
    def news_coverage(tickers: list[str]) -> dict[str, Any]:
        """Бумаги из списка, для которых есть таблица названий (covered), и те, что ищутся только
        по тикеру в тексте новости (uncovered)."""
        return guarded(lambda: service.coverage(tickers))

    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP-сервер новостей для дайджеста бота.")
    parser.add_argument("--db", required=True, help="Файл SQLite (абсолютный путь)")
    parser.add_argument("--feeds", nargs="*", default=[], help="Адреса лент (RSS/Atom, MOEX ISS)")
    parser.add_argument("--aliases", default=str(DEFAULT_ALIASES), help="Таблица алиасов (JSON)")
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    return parser.parse_args(argv)


def service_from_args(args: argparse.Namespace) -> NewsService:
    if not Path(args.db).is_absolute():
        raise SystemExit(
            "--db должен быть абсолютным путём (рабочий каталог процесса может меняться)."
        )
    if args.retention_days < 1:
        raise SystemExit("--retention-days должен быть не меньше 1.")
    table = nr.parse_alias_table(json.loads(Path(args.aliases).read_text(encoding="utf-8")))
    return NewsService(ns.NewsStore(args.db), args.feeds, table, args.retention_days)


def main() -> None:
    build_server(service_from_args(parse_args())).run("stdio")


if __name__ == "__main__":
    main()
