"""Детерминированная страховка поверх модели.

Модель хорошо парсит валидные запросы, но на «плохих» всё ещё иногда
придумывает успешный JSON. Guard срабатывает только в однозначных случаях
и не пытается угадывать bad_ticker/not_watch:

  - пустой ввод → empty;
  - есть watch-глагол/слова, нет тикера и нет «за …» → missing_tickers;
  - есть тикер, нет периода → missing_intervals;
  - есть тикер и ровно один период:
      * названа «сводка/отчёт» → missing_poll_interval,
      * иначе → missing_report_interval;
  - больше двух периодов → ambiguous.

Во всех остальных случаях возвращается ответ модели без изменений.
"""

from __future__ import annotations

import re

from spec import ALIAS_TO_TICKER, error, success

WATCH_WORDS = re.compile(
    r"\b(следи|следить|мониторь|мониторить|проверяй|проверять|наблюдай|наблюдать|"
    r"опрос|сводк|отч[её]т|итог)\w*",
    re.IGNORECASE,
)
REPORT_WORDS = re.compile(r"сводк|отч[её]т|итог", re.IGNORECASE)
NOT_WATCH_WORDS = re.compile(
    r"(выбрать|сравни|лучше|что думаешь|дивиденд|прогноз|расскажи|объясни|"
    r"сколько стоит|какие акции|посоветуй)",
    re.IGNORECASE,
)
PERIOD_RE = re.compile(
    r"(?:\d+\s*(?:минут\w*|час\w*|дн\w*|день|дня|дней|сут\w*|недел\w*|месяц\w*|год\w*|секунд\w*)"
    r"|полчас\w*|четверть часа|полсут\w*|двое суток|трое суток|два часа|три часа|"
    r"шесть часов|двенадцать часов|два дня|три дня|семь дней|час|сутки|неделю|день|дня|дней)",
    re.IGNORECASE,
)
CORRECTION_RE = re.compile(r"(нет,\s*не|хотя\s+нет|а\s+не\s+\d|лучше\s+(?:раз|сводк))", re.IGNORECASE)


def find_tickers(text: str) -> set[str]:
    lowered = text.lower().replace("ё", "е")
    found: set[str] = set()
    for alias, ticker in ALIAS_TO_TICKER.items():
        pattern = rf"(?<![a-zа-я0-9]){re.escape(alias)}(?![a-zа-я0-9])"
        if re.search(pattern, lowered):
            found.add(ticker)
    return found


def count_periods(text: str) -> int:
    return len(PERIOD_RE.findall(text))


def ticker_mentions(text: str) -> list[str]:
    """Все упоминания тикеров по порядку, включая повторы."""
    lowered = text.lower().replace("ё", "е")
    spans: list[tuple[int, int, str]] = []
    for alias, ticker in ALIAS_TO_TICKER.items():
        pattern = rf"(?<![a-zа-я0-9]){re.escape(alias)}(?![a-zа-я0-9])"
        for match in re.finditer(pattern, lowered):
            spans.append((match.start(), match.end(), ticker))
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[str] = []
    last_end = -1
    for start, end, ticker in spans:
        if start >= last_end:
            result.append(ticker)
            last_end = end
    return result


def period_value(text: str) -> str | None:
    value = text.lower().replace("ё", "е").strip()
    special = {
        "полчаса": "30m",
        "четверть часа": "15m",
        "полсуток": "12h",
        "двое суток": "2d",
        "трое суток": "3d",
        "два часа": "2h",
        "три часа": "3h",
        "шесть часов": "6h",
        "двенадцать часов": "12h",
        "два дня": "2d",
        "три дня": "3d",
        "семь дней": "7d",
        "час": "1h",
        "сутки": "1d",
        "день": "1d",
        "дня": "1d",
        "дней": "1d",
        "неделю": "7d",
    }
    if value in special:
        return special[value]
    match = re.match(
        r"(\d+)\s*(минут\w*|час\w*|дн\w*|день|дня|дней|сут\w*|недел\w*|месяц\w*|год\w*|секунд\w*)",
        value,
    )
    if not match:
        return None
    number = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("минут"):
        return f"{number}m"
    if unit.startswith("час"):
        return f"{number}h"
    if unit.startswith(("дн", "день", "дня", "дней", "сут")):
        return f"{number}d"
    return None


def period_values(text: str) -> list[str] | None:
    values = [period_value(match) for match in PERIOD_RE.findall(text)]
    if any(value is None for value in values):
        return None
    return values


def duplicate_ticker_success(user: str, output: dict) -> dict:
    """Если ambiguous вызван только дублем тикера, собираем успех по двум периодам."""
    if output.get("error") != "ambiguous":
        return output
    mentions = ticker_mentions(user)
    if len(mentions) < 2 or len(set(mentions)) == len(mentions):
        return output
    if len(REPORT_WORDS.findall(user)) != 1:
        return output
    periods = period_values(user)
    if periods is None or len(periods) != 2:
        return output
    matches = list(PERIOD_RE.finditer(user))
    report_match = REPORT_WORDS.search(user)
    if report_match is None:
        return output
    if report_match.start() <= matches[0].start():
        report, poll = periods[0], periods[1]
    elif report_match.start() <= matches[1].start():
        report, poll = periods[1], periods[0]
    else:
        return output
    return success(sorted(set(mentions)), poll, report)


def unknown_ticker_candidate(text: str) -> str | None:
    """Ищет название после «за»/«и»/запятой, которого нет в словаре тикеров.

    Сначала удаляем все известные алиасы (включая «Московской биржей»), затем
    смотрим только слова с заглавной буквы. «ценой»/«акциями» не срабатывают,
    «Ромашкой»/«TSLA» — срабатывают.
    """
    cleaned = text
    for alias in sorted(ALIAS_TO_TICKER, key=len, reverse=True):
        cleaned = re.sub(
            rf"(?<![A-Za-zА-Яа-я0-9]){re.escape(alias)}(?![A-Za-zА-Яа-я0-9])",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    match = re.search(r"(?:за|и|,)\s+([А-ЯЁA-Z][\w-]*)", cleaned)
    return match.group(1) if match else None


def apply_guard(user: str, output: dict | None) -> dict:
    if not user.strip():
        return error("empty")

    lowered = user.lower().replace("ё", "е")
    tickers = find_tickers(lowered)
    periods = count_periods(lowered)
    has_watch = bool(WATCH_WORDS.search(lowered))

    if unknown_ticker_candidate(user):
        return error("bad_ticker")

    if has_watch and not tickers and periods and "за " not in lowered:
        return error("missing_tickers")
    if tickers and periods == 0 and not has_watch and NOT_WATCH_WORDS.search(lowered):
        return error("not_watch")
    if has_watch and tickers and periods == 0:
        return error("missing_intervals")
    if has_watch and tickers and periods == 1:
        if REPORT_WORDS.search(lowered):
            return error("missing_poll_interval")
        return error("missing_report_interval")
    if periods > 2 and not CORRECTION_RE.search(lowered):
        return error("ambiguous")

    if not isinstance(output, dict):
        return error("empty")

    if output.get("ok") is not True:
        output = duplicate_ticker_success(user, output)

    # Согласованность тикеров: в успешном ответе должны быть ровно те тикеры,
    # которые явно названы в запросе. Это чинит и пропуск («SBER, SBER, GAZP»),
    # и лишний выдуманный тикер («Магнит» → MGNT, а не MTSS).
    if output.get("ok") is True and tickers:
        corrected = sorted(tickers)
        if output.get("secids") != corrected:
            output = {**output, "secids": corrected}

    return output
