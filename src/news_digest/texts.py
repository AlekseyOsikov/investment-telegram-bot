"""Тексты дайджеста новостей — чистые функции над данными сервера, без Telegram и сети.

Все строки пользователю — на русском, обычным текстом (без parse_mode). Оговорки — ФИКСИРОВАННЫЙ
текст кода (не модели) и входят в каждое сообщение: ответ `/news`, каждую часть дайджеста,
срочное сообщение (спека news-digest, требование «Оговорки в каждом сообщении»). Не убирать при
правке текстов. В фиксированных текстах нет оценок, гарантий и обещаний доходности.

Разбор (тип события, пересказ, механизм) приходит уже проверенным `rules.validate_analysis`;
если разбора нет — новость выводится с пометкой «разбор недоступен». Время — по Москве.
"""

from __future__ import annotations

from collections.abc import Sequence

from price_watch.texts import format_msk

from . import rules

# --- оговорки ------------------------------------------------------------------------------

DISCLAIMER_ADVISOR = "Бот не является лицензированным финансовым советником."
DISCLAIMER_NOT_ADVICE = "Это не инвестиционная рекомендация и не прогноз цены."
DISCLAIMER_COVERAGE = (
    "Подборка неполная: только общие ленты и новости биржи, без официального раскрытия эмитентов."
)
DISCLAIMER_AUTO = "Разбор выполнен автоматически и может быть неточным."
DISCLAIMERS = "\n".join(
    ("ℹ️ " + DISCLAIMER_ADVISOR, DISCLAIMER_NOT_ADVICE, DISCLAIMER_COVERAGE, DISCLAIMER_AUTO)
)

UNAVAILABLE_NOTE = "Разбор недоступен — показаны заголовок, источник и ссылка."
PRICE_NOTE_PREFIX = "Цена в это же время: "

NEED_WATCH_TEXT = (
    "Дайджест новостей строится по бумагам из опроса цен, а опрос в этом чате не задан. "
    "Сначала задай его командой /watch, например: /watch SBER GAZP 15m 1h — затем отправь /news."
)
NEWS_FAILED_TEXT = (
    "Не удалось включить дайджест новостей: сервер новостей сейчас недоступен. "
    "Попробуй позже; подробности записаны в журнал."
)

MAX_PART_CHARS = 3600  # запас от TELEGRAM_MESSAGE_LIMIT на заголовок и оговорки


def _plural(number: int, forms: tuple[str, str, str]) -> str:
    one, few, many = forms
    if number % 10 == 1 and number % 100 != 11:
        return one
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return few
    return many


def _every_hours(hours: int) -> str:
    if hours == 1:
        return "раз в час"
    return f"раз в {hours} {_plural(hours, ('час', 'часа', 'часов'))}"


def _coverage_limited_line(uncovered: Sequence[str]) -> str:
    return (
        f"⚠️ Покрытие ограничено по бумагам {', '.join(uncovered)}: для них нет таблицы "
        "названий, новости ищутся только по тикеру в тексте."
    )


# --- команды -------------------------------------------------------------------------------


def news_enabled_text(
    tickers: Sequence[str],
    digest_interval_hours: int,
    max_messages_per_day: int,
    uncovered: Sequence[str] = (),
) -> str:
    lines = [
        "📰 Дайджест новостей включён.",
        f"Бумаги (из опроса /watch): {', '.join(tickers)}.",
        f"Присылаю дайджест {_every_hours(digest_interval_hours)}, только если появились "
        f"новые новости; важные могут прийти отдельным сообщением. Не более "
        f"{max_messages_per_day} сообщений в сутки.",
        "К каждой новости — тип события и общий механизм, через который такие события обычно "
        "связаны с ценой; без направления и прогноза.",
    ]
    if uncovered:
        lines.append(_coverage_limited_line(uncovered))
    lines += ["", DISCLAIMERS, "", "Остановить: /news_stop, состояние: /news_status."]
    return "\n".join(lines)


def news_status_text(
    enabled: bool,
    tickers: Sequence[str],
    next_digest_iso: str | None,
    messages_today: int,
    max_messages_per_day: int,
    uncovered: Sequence[str] = (),
) -> str:
    if not enabled:
        return "Дайджест новостей выключен. Включить: /news (нужен опрос цен /watch)."
    lines = [
        "📰 Дайджест новостей включён.",
        "Бумаги: "
        + (", ".join(tickers) if tickers else "опрос цен не задан — дайджест остановится")
        + ".",
        f"Ближайший дайджест не раньше: {format_msk(next_digest_iso)}.",
        f"Сообщений сегодня: {messages_today} из {max_messages_per_day}.",
    ]
    if uncovered:
        lines.append(_coverage_limited_line(uncovered))
    lines += ["", DISCLAIMER_ADVISOR + " " + DISCLAIMER_NOT_ADVICE]
    return "\n".join(lines)


def news_stop_text(was_enabled: bool) -> str:
    if was_enabled:
        return (
            "Дайджест новостей выключен, отметки дайджеста этого чата удалены. "
            "Опрос цен продолжает работать (остановить: /watch_stop)."
        )
    return "Дайджест новостей и так не был включён."


# --- сообщения дайджеста -------------------------------------------------------------------


def _tickers_label(item: dict) -> str:
    return ", ".join(entry["ticker"] for entry in item.get("tickers") or []) or "—"


def item_block(item: dict, price_line: str | None = None) -> str:
    """Один элемент: заголовок, тип и важность, пересказ, механизм, цена (если есть), источник."""
    lines = [f"• [{_tickers_label(item)}] {item.get('title', '').strip()}"]
    analysis = item.get("analysis")
    if analysis:
        event = rules.EVENT_TYPES.get(analysis.get("event_type"), "Прочее")
        level = rules.IMPORTANCE_LEVELS.get(analysis.get("importance"), "—")
        lines.append(f"  Тип события: {event} · важность: {level}")
        lines.append(f"  О чём: {analysis['summary']}")
        lines.append(
            f"  Как такие события связаны с ценой (общее правило): {analysis['mechanism']}"
        )
    else:
        lines.append(f"  {UNAVAILABLE_NOTE}")
    if price_line:
        lines.append(f"  {price_line}")
    published = format_msk(item.get("published_at"), with_zone=False)
    lines.append(f"  {item.get('source', '')} · {published} МСК · {item.get('link', '')}")
    return "\n".join(lines)


def digest_messages(
    items: Sequence[dict],
    price_lines: dict[str, str] | None = None,
    limit: int = MAX_PART_CHARS,
) -> list[str]:
    """Дайджест как список сообщений не длиннее `limit`; ОТДЕЛЬНЫЕ части получают собственный
    заголовок и оговорки (каждое сообщение самодостаточно). Пустой список новостей → []."""
    if not items:
        return []
    price_lines = price_lines or {}
    blocks = [item_block(item, price_lines.get(item.get("id", ""))) for item in items]
    reserve = len(DISCLAIMERS) + 120
    parts: list[list[str]] = [[]]
    size = 0
    for block in blocks:
        block = block[: max(limit - reserve, 200)]
        if parts[-1] and size + len(block) + 2 > limit - reserve:
            parts.append([])
            size = 0
        parts[-1].append(block)
        size += len(block) + 2
    total = len(parts)
    messages = []
    for index, blocks_in_part in enumerate(parts, start=1):
        suffix = f" (часть {index} из {total})" if total > 1 else ""
        header = f"📰 Новости по твоим бумагам: {len(items)} шт.{suffix}"
        messages.append("\n\n".join([header, *blocks_in_part, DISCLAIMERS]))
    return messages


def fit_telegram(text: str, limit: int = 4000) -> str:
    """Срочное сообщение об одной новости обычно короче лимита Telegram; на случай аномально
    длинного текста он усекается, а оговорки остаются в конце."""
    if len(text) <= limit:
        return text
    head = text[: limit - len(DISCLAIMERS) - 4].rstrip()
    return f"{head}…\n\n{DISCLAIMERS}"


def urgent_message(item: dict, price_line: str | None = None) -> str:
    """Срочное сообщение об одной новости высокой важности."""
    return "\n\n".join(
        ["❗ Важная новость по твоей бумаге", item_block(item, price_line), DISCLAIMERS]
    )
