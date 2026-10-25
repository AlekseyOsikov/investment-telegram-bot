"""Тексты команд опроса цен — чистые функции над ответами сервера, без Telegram и сети.

Все строки пользователю — на русском, обычным текстом (без parse_mode): тикеры, имена
и `<тикеры>` в подсказке не нужно экранировать. Времена приходят от сервера строками
ISO 8601 со смещением +03:00 и показываются как есть по московскому времени — пояс
пользователя Telegram боту не сообщает.

Текст самой сводки (`text` из `watch_run_due`/`watch_get_report`) здесь НЕ формируется:
его собирает сервер, и бот отправляет его без правок, чтобы оговорки сервера
(задержка котировок, «не инвестиционная рекомендация») сохранялись — см. требование
«Оговорки и осведомлённость пользователя» в спеке price-watch.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

MSK = timezone(timedelta(hours=3))

DELAY_NOTE = "Котировки акций, облигаций и фондов задержаны на 15 минут (индексы без задержки)."

USAGE_TEXT = (
    "Формат: /watch <тикеры> <период опроса> <период сводки>\n"
    "Пример: /watch SBER GAZP 15m 1h — опрашивать цены SBER и GAZP каждые 15 минут и "
    "присылать сводку раз в час.\n\n"
    "Период — число и единица: m или м (минуты), h или ч (часы), d или д (сутки), "
    "например 15m, 1h, 1д. Тикеры — через пробел или запятую. Новая команда заменяет "
    "прежний опрос этого чата.\n\n"
    "Ещё: /watch_status — состояние опроса, /watch_report — сводка прямо сейчас, "
    "/watch_stop — остановить опрос."
)

NO_WATCH_TEXT = (
    "Опрос цен не задан. Задай его командой /watch, например: /watch SBER GAZP 15m 1h"
)

_INTERVAL_RE = re.compile(r"^(\d+)([mhd])$")
# Формы слова для «раз в N …»: (1 → винительный, 2-4, 5+).
_UNIT_FORMS = {
    "m": ("минуту", "минуты", "минут"),
    "h": ("час", "часа", "часов"),
    "d": ("день", "дня", "дней"),
}


def _plural_form(number: int, forms: tuple[str, str, str]) -> str:
    one, few, many = forms
    if number % 10 == 1 and number % 100 != 11:
        return one
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return few
    return many


def every_text(interval: str) -> str:
    """`15m` → «раз в 15 минут», `1h` → «раз в час»; неразобранное значение как есть."""
    match = _INTERVAL_RE.match(interval)
    if match is None:
        return interval
    number, unit = int(match.group(1)), match.group(2)
    word = _plural_form(number, _UNIT_FORMS[unit])
    return f"раз в {word}" if number == 1 else f"раз в {number} {word}"


def format_msk(iso: str | None, *, with_zone: bool = True) -> str:
    """ISO-время сервера → `ДД.ММ ЧЧ:ММ (МСК)`; нераспознанное значение возвращается как есть."""
    if not iso:
        return "—"
    try:
        moment = datetime.fromisoformat(iso).astimezone(MSK)
    except ValueError:
        return iso
    text = moment.strftime("%d.%m %H:%M")
    return f"{text} (МСК)" if with_zone else text


def format_price(value: float) -> str:
    """Цена с запятой и без хвоста нулей: 301.2 → «301,2», 100.0 → «100»."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _unit_text(unit: str | None) -> str:
    if unit == "percent_of_face":
        return "% от номинала"
    if unit == "points":
        return "пунктов"
    return unit or ""


def _sample_line(sample: dict) -> str:
    price = format_price(sample["price"])
    unit = _unit_text(sample.get("price_unit"))
    quote = format_msk(sample.get("quote_at"), with_zone=False)
    return f"• {sample['secid']}: {price} {unit}".rstrip() + f" (котировка на {quote})"


def friendly_tool_error(text: str) -> str:
    """Текст отказа сервера → текст для пользователя.

    Обычные отказы сервера (границы периодов, неизвестный тикер) уже написаны по-русски и
    проходят как есть. Отказ входной СХЕМЫ приходит сырым английским дампом валидатора
    («1 validation error for ...») — его заменяем общей фразой; подробности остаются в
    журнале (пишет вызывающий код).
    """
    if "validation error" in text:
        return (
            "сервер не принял параметры команды: проверь тикеры и периоды "
            "(формат: /watch SBER GAZP 15m 1h)."
        )
    return text


def watch_set_text(result: dict) -> str:
    """Ответ на успешную `/watch` по результату `watch_set`."""
    head = "✅ Опрос задан" + (" (прежний опрос заменён)" if result.get("replaced") else "")
    lines = [
        head,
        f"Бумаги: {', '.join(result['secids'])}",
        f"Опрос: {every_text(result['poll_interval'])}, "
        f"сводка: {every_text(result['report_interval'])}",
        f"Следующий опрос: {format_msk(result.get('next_poll_at'))}",
        f"Следующая сводка: {format_msk(result.get('next_report_at'))}",
    ]
    samples = result.get("first_samples") or []
    if samples:
        lines += ["", "Первые цены:"] + [_sample_line(sample) for sample in samples]
    lines += ["", DELAY_NOTE]
    return "\n".join(lines)


def watch_status_text(result: dict) -> str:
    """Ответ на `/watch_status` по результату `watch_status`."""
    if not result.get("active"):
        return NO_WATCH_TEXT
    lines = [
        "📡 Опрос цен",
        f"Бумаги: {', '.join(result['secids'])}",
        f"Опрос: {every_text(result['poll_interval'])}, "
        f"сводка: {every_text(result['report_interval'])}",
        f"Задан: {format_msk(result.get('started_at'))}",
        f"Следующий опрос: {format_msk(result.get('next_poll_at'))}",
        f"Следующая сводка: {format_msk(result.get('next_report_at'))}",
        f"Замеров в текущем периоде: {result.get('samples_in_period', 0)}",
    ]
    pending = result.get("pending_reports") or 0
    if pending:
        lines.append(f"Сводок, ожидающих доставки: {pending}")
    return "\n".join(lines)


def watch_stop_text(result: dict) -> str:
    """Ответ на `/watch_stop` по результату `watch_stop`."""
    if result.get("stopped"):
        return "🛑 Опрос остановлен, накопленные замеры удалены."
    return "Опроса и не было: останавливать нечего."
