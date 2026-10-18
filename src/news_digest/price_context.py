"""Совпадение по времени с движением цены (спека news-digest; design.md изменения add-news-digest,
решение 7): чистая функция изменения цены между закрытием последней свечи до публикации новости
и последней известной ценой и формулировка строки для дайджеста.

Строка называет ТОЛЬКО совпадение по времени: слов о причине нет (проверяет тест), есть пометка о
задержке котировок и что это изменение цены, а не доходность. Если данных нет, интервал слишком
короткий или новость старше окна — строки нет вовсе (`None`). Свечи — как у `get_price_history`
сервера mcp-moex (`interval=hour`): `begin` в ISO 8601 с московским смещением.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from price_watch.texts import MSK

CANDLE_MINUTES = 60  # размер свечи запроса (interval=hour)
MAX_NEWS_AGE_HOURS = 24  # старше — совпадение по времени уже ничего не говорит
MIN_SPAN_MINUTES = 60  # короче — шум, строку не показываем


@dataclass(frozen=True)
class PriceChange:
    pct: float
    start_price: float
    end_price: float
    start_at: datetime  # конец последней свечи до публикации (момент базовой цены)
    end_at: datetime  # конец последней известной свечи


def _moment(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=MSK)


def _prices(candle: dict) -> tuple[datetime, float, float] | None:
    begin = _moment(candle.get("begin"))
    try:
        close = float(candle["close"])
        open_ = float(candle["open"])
    except (KeyError, TypeError, ValueError):
        return None
    return (begin, open_, close) if begin is not None else None


def price_change_since(
    candles: list[dict] | None,
    published_at: datetime,
    now: datetime,
    *,
    max_age_hours: int = MAX_NEWS_AGE_HOURS,
    min_span_minutes: int = MIN_SPAN_MINUTES,
) -> PriceChange | None:
    """Изменение цены между закрытием последней свечи, завершившейся до публикации, и закрытием
    последней свечи. None — нет данных, новость старше окна или промежуток слишком короткий."""
    if not candles:
        return None
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    if now - published_at > timedelta(hours=max_age_hours) or published_at > now:
        return None

    parsed = sorted((p for p in map(_prices, candles) if p), key=lambda p: p[0])
    candle_length = timedelta(minutes=CANDLE_MINUTES)
    before = [p for p in parsed if p[0] + candle_length <= published_at]
    if not before or not parsed:
        return None
    base_begin, _, base_close = before[-1]
    last_begin, _, last_close = parsed[-1]
    start_at, end_at = base_begin + candle_length, last_begin + candle_length
    if base_close <= 0 or end_at - start_at < timedelta(minutes=min_span_minutes):
        return None
    return PriceChange(
        pct=(last_close - base_close) / base_close * 100,
        start_price=base_close,
        end_price=last_close,
        start_at=start_at,
        end_at=end_at,
    )


def _signed_pct(value: float) -> str:
    text = f"{abs(value):.2f}".rstrip("0").rstrip(".") or "0"
    sign = "+" if value > 0 else "−" if value < 0 and text != "0" else ""
    return f"{sign}{text.replace('.', ',')}%"


def coincidence_line(change: PriceChange | None) -> str | None:
    """Строка для дайджеста; без данных — None."""
    if change is None:
        return None
    start = change.start_at.astimezone(MSK).strftime("%d.%m %H:%M")
    end = change.end_at.astimezone(MSK).strftime("%d.%m %H:%M")
    return (
        f"Цена в это же время: с {start} до {end} (МСК) изменилась на {_signed_pct(change.pct)}. "
        "Это совпадение по времени, связь с новостью не установлена; котировки задержаны на "
        "15 минут; изменение цены — не доходность."
    )
