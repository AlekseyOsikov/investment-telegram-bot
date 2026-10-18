"""Разбор и получение лент сервера новостей (спека news-server, требование «Источники»):
RSS/Atom и ответ MOEX ISS `sitenews` приводятся к одному виду `FeedItem`.

Формат ответа определяется по содержимому (`{` — JSON биржи, иначе XML), поэтому адреса лент
задаёт оператор одним списком. Сеть только в `fetch_url` (тайм-аут, предел размера, без ключей);
остальное — чистые функции, проверяемые на фиксированных образцах без сети
(`tests/test_news_server_feeds.py`). Отказ в доступе (401/403/429) НЕ обходится: лента попадает
в отчёт сбора как недоступная. Не часть бота: `src/` этот модуль не импортирует.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import news_rules as nr

USER_AGENT = "investment-telegram-bot-news/1.0 (+public feeds, no auth)"
DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_BYTES = 2_000_000
# Ссылка на новость биржи по идентификатору из ответа ISS.
MOEX_NEWS_URL = "https://www.moex.com/n{id}"

ATOM = "{http://www.w3.org/2005/Atom}"


class FeedError(Exception):
    """Лента не получена или не разобрана; текст понятен оператору и пользователю."""


@dataclass(frozen=True)
class FeedItem:
    source: str
    title: str
    link: str
    summary: str
    published_at: datetime | None  # с часовым поясом; None — лента время не указала


@dataclass(frozen=True)
class FeedReport:
    url: str
    ok: bool
    count: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return {"url": self.url, "ok": self.ok, "count": self.count, "error": self.error}


def source_name(url: str) -> str:
    """Источник для показа — хост ленты без `www.`."""
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host or url


def _parse_datetime(value: str | None, *, naive_tz: timezone = timezone.utc) -> datetime | None:
    if not value or not value.strip():
        return None
    text = value.strip()
    try:
        moment = parsedate_to_datetime(text)  # RFC 822 (RSS)
    except (TypeError, ValueError):
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))  # ISO (Atom, ISS)
        except ValueError:
            return None
    return moment if moment.tzinfo else moment.replace(tzinfo=naive_tz)


def parse_iss_sitenews(body: str, source: str) -> list[FeedItem]:
    """Ответ `sitenews.json` биржи: колонки id, title, published_at (московское время)."""
    try:
        data = json.loads(body)
        block = data["sitenews"]
        columns = block["columns"]
        rows = block["data"]
    except (ValueError, KeyError, TypeError) as exc:
        raise FeedError("ответ биржи не похож на список новостей") from exc
    index = {name: i for i, name in enumerate(columns)}
    if "id" not in index or "title" not in index:
        raise FeedError("в ответе биржи нет полей id и title")
    items: list[FeedItem] = []
    for row in rows:
        title = nr.normalize_text(str(row[index["title"]]))
        if not title:
            continue
        published = (
            _parse_datetime(row[index["published_at"]], naive_tz=nr.MSK)
            if "published_at" in index
            else None
        )
        items.append(
            FeedItem(source, title, MOEX_NEWS_URL.format(id=row[index["id"]]), "", published)
        )
    return items


def _text(element: ET.Element | None) -> str:
    return nr.normalize_text(element.text if element is not None else "")


def parse_xml_feed(body: str, source: str) -> list[FeedItem]:
    """RSS 2.0 (`item`) и Atom (`entry`)."""
    # Объявления типа документа и сущности в публичных лентах не нужны; отказ защищает
    # от расширения сущностей, а не от обычных лент.
    if "<!DOCTYPE" in body.upper() or "<!ENTITY" in body.upper():
        raise FeedError("лента содержит объявление типа документа и отклонена")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise FeedError("лента не является корректным XML") from exc

    items: list[FeedItem] = []
    for node in root.iter("item"):  # RSS
        title = _text(node.find("title"))
        if title:
            items.append(
                FeedItem(
                    source,
                    title,
                    _text(node.find("link")),
                    _text(node.find("description")),
                    _parse_datetime(_text(node.find("pubDate"))),
                )
            )
    for node in root.iter(f"{ATOM}entry"):  # Atom
        title = _text(node.find(f"{ATOM}title"))
        if not title:
            continue
        link_node = node.find(f"{ATOM}link")
        link = (link_node.get("href") or "") if link_node is not None else ""
        summary = _text(node.find(f"{ATOM}summary")) or _text(node.find(f"{ATOM}content"))
        stamp = _text(node.find(f"{ATOM}published")) or _text(node.find(f"{ATOM}updated"))
        items.append(FeedItem(source, title, link.strip(), summary, _parse_datetime(stamp)))
    return items


def parse_feed(body: str, source: str) -> list[FeedItem]:
    """Определяет формат по содержимому и разбирает ленту."""
    if body.lstrip().startswith("{"):
        return parse_iss_sitenews(body, source)
    return parse_xml_feed(body, source)


def fetch_url(
    url: str, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = DEFAULT_MAX_BYTES
) -> str:
    """Скачивает ленту: тайм-аут, предел размера, без ключей и без обхода отказов в доступе."""
    if urlsplit(url).scheme not in ("http", "https"):
        raise FeedError("адрес ленты должен начинаться с http:// или https://")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(max_bytes + 1)
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 429):
            raise FeedError(f"доступ закрыт для автоматических запросов (HTTP {exc.code})") from exc
        raise FeedError(f"сервер ответил HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeedError(f"лента не отвечает ({type(exc).__name__})") from exc
    if len(raw) > max_bytes:
        raise FeedError("ответ ленты слишком большой")
    return raw.decode(charset, errors="replace")


Fetch = Callable[[str], str]


def collect_feeds(urls: list[str], fetch: Fetch) -> tuple[list[FeedItem], list[FeedReport]]:
    """Забирает все ленты; сбой одной не ломает остальные и попадает в отчёт."""
    items: list[FeedItem] = []
    reports: list[FeedReport] = []
    for url in urls:
        try:
            parsed = parse_feed(fetch(url), source_name(url))
        except FeedError as exc:
            reports.append(FeedReport(url, False, error=str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 — любая ошибка одной ленты не должна ломать сбор
            reports.append(
                FeedReport(url, False, error=f"неожиданная ошибка ({type(exc).__name__})")
            )
            continue
        items.extend(parsed)
        reports.append(FeedReport(url, True, count=len(parsed)))
    return items, reports
