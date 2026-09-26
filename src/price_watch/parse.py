"""Разбор аргументов команды /watch — чистые функции, без Telegram и без сети.

Формат: `/watch <тикеры> <период опроса> <период сводки>`, например
`/watch SBER GAZP 15m 1h`. Два последних слова — периоды, всё перед ними — тикеры
(через пробелы и/или запятые). Период — целое число и единица: `m`/`h`/`d` либо
русские `м`/`ч`/`д` (минуты, часы, сутки), регистр не важен; русские единицы
переводятся в латинские, потому что сервер понимает только их.

Бот проверяет ТОЛЬКО форму. Границы периодов, их взаимное соотношение, число тикеров
и существование бумаги проверяет сервер, и его отказ передаётся пользователю как есть
(price_watch/commands.py): правила лежат в одном месте и не могут разойтись.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INTERVAL_RE = re.compile(r"^(\d+)([mhdмчд])$", re.IGNORECASE)
_UNITS = {"m": "m", "h": "h", "d": "d", "м": "m", "ч": "h", "д": "d"}
# Форма тикера — как у входной схемы сервера (латиница, цифры, `.`, `_`, `-`, до 40
# символов). Проверяется здесь, потому что иначе отказ схемы приходит пользователю
# сырым английским дампом валидатора (в живой проверке так вышло на кириллице).
_TICKER_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")


@dataclass(frozen=True)
class WatchRequest:
    """Разобранная команда: тикеры как ввёл пользователь и периоды в формате сервера."""

    secids: list[str]
    poll_interval: str
    report_interval: str


@dataclass(frozen=True)
class ParseError:
    """Ошибка формы. reason — что не так (None для пустой команды: тогда достаточно
    одной подсказки с форматом)."""

    reason: str | None


def normalize_interval(word: str) -> str | None:
    """`15м` → `15m`, `1H` → `1h`; None, если слово не похоже на период."""
    match = _INTERVAL_RE.match(word.strip())
    if match is None:
        return None
    return f"{match.group(1)}{_UNITS[match.group(2).lower()]}"


def parse_watch_args(args: list[str]) -> WatchRequest | ParseError:
    """Аргументы команды (слова после `/watch`) → запрос или ошибка формы."""
    words = [word for word in args if word.strip()]
    if not words:
        return ParseError(None)

    if len(words) < 2:
        return ParseError("Не хватает периодов: после тикеров нужны период опроса и период сводки.")

    report_interval = normalize_interval(words[-1])
    poll_interval = normalize_interval(words[-2])
    if report_interval is None or poll_interval is None:
        return ParseError(
            "Два последних слова команды должны быть периодами (период опроса и период "
            "сводки), например 15m и 1h."
        )

    secids = [
        ticker
        for word in words[:-2]
        for ticker in word.split(",")
        if ticker.strip()
    ]
    if not secids:
        return ParseError("Не указаны тикеры: они идут перед периодами.")

    misplaced = [ticker for ticker in secids if normalize_interval(ticker) is not None]
    if misplaced:
        return ParseError(
            f"«{misplaced[0]}» похоже на период, а не на тикер: периоды пишутся в конце "
            "команды, после всех тикеров."
        )

    secids = [ticker.strip() for ticker in secids]
    malformed = [ticker for ticker in secids if _TICKER_RE.match(ticker) is None]
    if malformed:
        return ParseError(
            f"«{malformed[0]}» не похоже на тикер: тикер состоит из латинских букв, цифр и "
            "знаков . _ - (например, SBER или SU26238RMFS4)."
        )

    return WatchRequest(
        secids=secids,
        poll_interval=poll_interval,
        report_interval=report_interval,
    )
