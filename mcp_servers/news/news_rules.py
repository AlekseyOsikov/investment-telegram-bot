"""Чистые правила сервера новостей (спека news-server; design.md изменения add-news-digest,
решение 3): устойчивый идентификатор новости, нормализация текста, привязка к тикерам по
таблице алиасов, усечение фрагмента. Без сети, диска и времени: всё проверяется тестами
(`tests/test_news_server_rules.py`).

Привязку выполняет КОД, а не модель: новость, у которой код не нашёл бумаги, моделью не
разбирается и в дайджест не попадает. Каждая привязка называет совпадение (по чему найдена).
Не часть бота: `src/` этот модуль не импортирует.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MSK = timezone(timedelta(hours=3))

# Сколько символов фрагмента текста новости хранится (полный текст статей не хранится).
DEFAULT_EXCERPT_CHARS = 600

# Способ привязки в записи: по тикеру латиницей, по названию, по короткому алиасу с контекстом.
KIND_TICKER = "ticker"
KIND_NAME = "name"
KIND_NAME_CONTEXT = "name+context"

# Окончание русского слова, допустимое после алиаса («Сбербанка», «Газпромом»).
_MAX_SUFFIX = 3
_TRACKING_PARAMS = ("utm_", "yclid", "fbclid", "gclid", "from", "ref")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Убирает HTML-теги и сущности, схлопывает пробелы. Пустое значение → пустая строка."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _SPACE_RE.sub(" ", text).strip()


def truncate_excerpt(text: str | None, limit: int = DEFAULT_EXCERPT_CHARS) -> str:
    """Фрагмент текста не длиннее `limit` символов, по границе слова, с многоточием."""
    clean = normalize_text(text)
    if limit < 1:
        return ""
    if len(clean) <= limit:
        return clean
    cut = clean[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-—") + "…"


def canonical_link(link: str | None) -> str:
    """Ссылка без якоря и меток отслеживания, схема и хост в нижнем регистре. Не ссылка → ''."""
    if not link:
        return ""
    parts = urlsplit(link.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return ""
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_PARAMS)
    ]
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(query), "")
    )


def news_id(link: str | None, title: str | None, source: str | None) -> str:
    """Устойчивый идентификатор: по ссылке, а без неё — по заголовку и источнику. Одна
    новость в двух лентах (одна ссылка) даёт один идентификатор."""
    canonical = canonical_link(link)
    if canonical:
        basis = "link:" + canonical
    else:
        basis = "title:" + normalize_text(title).lower() + "|" + (source or "").strip().lower()
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


def msk_date(moment: datetime) -> str:
    """Московские сутки момента (`ГГГГ-ММ-ДД`): по ним считается суточный лимит сообщений."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(MSK).strftime("%Y-%m-%d")


# --- таблица алиасов ---------------------------------------------------------------------


@dataclass(frozen=True)
class WeakAlias:
    """Короткий или двусмысленный алиас: засчитывается, только если рядом в тексте есть одно
    из слов-контекстов (основы слов, регистр не важен)."""

    alias: str
    context: tuple[str, ...]


@dataclass(frozen=True)
class AliasEntry:
    names: tuple[str, ...] = ()
    weak: tuple[WeakAlias, ...] = ()
    # Фразы, с которых НЕ должно начинаться совпадение алиаса («Газпром нефть» не «Газпром»).
    excludes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TickerMatch:
    ticker: str
    matched: str  # то, по чему найдена привязка (слово из текста или алиас)
    kind: str


def parse_alias_table(data: object) -> dict[str, AliasEntry]:
    """Таблица алиасов из JSON: `{"TICKER": {"names": [...], "weak": [{"alias": "...",
    "context": [...]}], "excludes": [...]}}`. Нарушение схемы — ValueError с понятным текстом."""
    if not isinstance(data, dict):
        raise ValueError("Таблица алиасов должна быть объектом «тикер → описание».")
    table: dict[str, AliasEntry] = {}
    for ticker, raw in data.items():
        if ticker.startswith("_"):  # служебные поля вроде «_comment»
            continue
        if not re.fullmatch(r"[A-Z0-9]{2,12}", ticker):
            raise ValueError(f"Тикер {ticker!r}: нужны заглавные латинские буквы и цифры.")
        if not isinstance(raw, dict):
            raise ValueError(f"Тикер {ticker}: описание должно быть объектом.")
        names = _string_list(raw.get("names", []), f"{ticker}.names")
        excludes = _string_list(raw.get("excludes", []), f"{ticker}.excludes")
        weak_raw = raw.get("weak", [])
        if not isinstance(weak_raw, list):
            raise ValueError(f"{ticker}.weak: нужен список.")
        weak: list[WeakAlias] = []
        for item in weak_raw:
            if not isinstance(item, dict) or not isinstance(item.get("alias"), str):
                raise ValueError(f"{ticker}.weak: элемент должен быть {{alias, context}}.")
            context = _string_list(item.get("context", []), f"{ticker}.weak.context")
            if not context:
                raise ValueError(
                    f"{ticker}: короткому алиасу {item['alias']!r} нужен непустой context."
                )
            weak.append(WeakAlias(item["alias"].strip(), tuple(context)))
        if not names and not weak:
            raise ValueError(f"Тикер {ticker}: нет ни одного названия.")
        table[ticker] = AliasEntry(tuple(names), tuple(weak), tuple(excludes))
    return table


def _string_list(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{where}: нужен список непустых строк.")
    return [v.strip() for v in value]


def _alias_regex(alias: str) -> re.Pattern[str]:
    suffix = rf"\w{{0,{_MAX_SUFFIX}}}" if re.search(r"[А-Яа-яЁё]$", alias) else ""
    return re.compile(rf"(?<!\w){re.escape(alias)}{suffix}(?!\w)", re.IGNORECASE)


def _starts_with_exclude(text: str, position: int, excludes: tuple[str, ...]) -> bool:
    tail = text[position:].lower()
    return any(tail.startswith(phrase.lower()) for phrase in excludes)


def _has_context(text: str, context: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"(?<!\w){re.escape(stem.lower())}", lowered) for stem in context)


def match_ticker(text: str, ticker: str, entry: AliasEntry) -> TickerMatch | None:
    """Привязка текста к ОДНОЙ бумаге или None. Приоритет: тикер латиницей (целое слово,
    заглавными) → название → короткий алиас с контекстом."""
    ticker_match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(ticker)}(?![A-Za-z0-9_])", text)
    if ticker_match:
        return TickerMatch(ticker, ticker_match.group(0), KIND_TICKER)
    for name in entry.names:
        for found in _alias_regex(name).finditer(text):
            if not _starts_with_exclude(text, found.start(), entry.excludes):
                return TickerMatch(ticker, found.group(0), KIND_NAME)
    for weak in entry.weak:
        for found in _alias_regex(weak.alias).finditer(text):
            if _starts_with_exclude(text, found.start(), entry.excludes):
                continue
            if _has_context(text, weak.context):
                return TickerMatch(ticker, found.group(0), KIND_NAME_CONTEXT)
    return None


def match_tickers(
    title: str | None,
    body: str | None,
    table: dict[str, AliasEntry],
    extra_tickers: tuple[str, ...] = (),
) -> list[TickerMatch]:
    """Все бумаги, к которым относится новость, по заголовку и тексту. Тикер, которого нет
    в таблице алиасов (`extra_tickers` — наблюдаемые чатами), ищется только как слово
    латиницей в верхнем регистре."""
    text = normalize_text(f"{title or ''}. {body or ''}")
    matches: list[TickerMatch] = []
    for ticker, entry in table.items():
        found = match_ticker(text, ticker, entry)
        if found:
            matches.append(found)
    for ticker in extra_tickers:
        if ticker in table:
            continue
        found = match_ticker(text, ticker, AliasEntry(names=(), weak=()))
        if found:
            matches.append(found)
    return matches
