"""Чистые правила дайджеста новостей: перечни типов событий и важности, терпимый разбор JSON из
ответа модели, проверка разбора КОДОМ (границы допустимого записаны здесь, а не только в
промпте), решение «повторить / откатиться», выбор элементов дайджеста и срочных, сроки.

Граница (design.md изменения add-news-digest, решение 4): допустимы тип события, важность, одна
фраза пересказа и общий механизм типа события; недопустимы направление влияния, прогноз, оценка
«хорошая/плохая новость», советы. Модуль не знает ни про Telegram, ни про сеть, ни про модель —
тесты `tests/test_news_digest_rules.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Типы событий: код (его возвращает модель) → подпись для пользователя.
EVENT_TYPES: dict[str, str] = {
    "dividends": "Дивиденды",
    "reporting": "Отчётность",
    "sanctions": "Санкции и регуляторные меры",
    "mna": "Слияния и поглощения",
    "management": "Смена руководства",
    "deals": "Сделки и контракты",
    "disputes": "Судебные и иные споры",
    "exchange": "Меры биржи",
    "other": "Прочее",
}
IMPORTANCE_LEVELS: dict[str, str] = {"high": "высокая", "medium": "средняя", "low": "низкая"}
IMPORTANCE_RANK: dict[str, int] = {"high": 0, "medium": 1, "low": 2}

MAX_SUMMARY_CHARS = 300
MAX_MECHANISM_CHARS = 300
MAX_ATTEMPTS = 2  # первый ответ и один повтор при нарушении

# Запретные формулировки: регулярные выражения по основам слов, регистр не важен. Единый
# перечень — его проверяют и бот, и оценка (research/news_eval.py). Проверка строгая:
# ложное срабатывание ведёт к показу новости без разбора, пропуск — к нарушению границы.
_DIRECTION = (
    r"позитив\w*",
    r"негатив\w*",
    r"благоприятн\w*",
    r"неблагоприятн\w*",
    r"быч\w+",
    r"медвеж\w+",
    r"рост(?:а|у|ом|е|ы|ов)?",
    r"(?:вы|по|под)?раст(?:ёт|ет|ут|и|ёшь)",
    r"(?:вы|под|по)рос\w*",
    r"подорожа\w*",
    r"дорожа\w*",
    r"подешев\w*",
    r"дешевеет\w*",
    r"паден\w+",
    r"пада\w+",
    r"упад\w+",
    r"упас\w*",
    r"обвал\w*",
    r"взлёт\w*",
    r"взлет\w*",
    r"просад\w*",
    r"отскок\w*",
    r"снижен\w+\s+(?:цен\w*|котировок|курса|стоимост\w*)",
    r"(?:хорош|плох)(?:ая|ой|ие|ую|ей)\s+новост\w*",
)
_ADVICE = (
    r"покупа\w*",
    r"купи\w*",
    r"покупк\w*",
    r"продава\w*",
    r"прода(?:ть|йте|жа|жу)",
    r"стоит",
    r"рекоменду\w+",
    r"рекомендаци\w+",
    r"совету\w+",
    r"держ(?:ать|ите)",
    r"фиксир\w+\s+прибыл\w+",
    r"лонг\w*",
    r"шорт\w*",
    r"сигнал\w*",
    r"целев\w+\s+цен\w*",
)
_FORECAST = (
    r"ожида\w+",
    r"прогноз\w*",
    r"предполага\w+",
    r"скорее\s+всего",
    r"вероятн\w*",
    r"наверняка",
    r"гарантир\w+",
    r"гарантирован\w+",
    r"повлия\w+",
    r"приведёт",
    r"приведет",
    r"станет\s+причиной",
    r"будет\s+(?:расти|падать|снижаться|дорожать|дешеветь|повышаться)",
)
FORBIDDEN_PATTERNS: tuple[str, ...] = _DIRECTION + _ADVICE + _FORECAST
_FORBIDDEN_RE = re.compile(r"(?<!\w)(?:" + "|".join(FORBIDDEN_PATTERNS) + r")(?!\w)", re.IGNORECASE)


def find_forbidden(text: str) -> list[str]:
    """Запретные формулировки (направление, прогноз, совет, оценка), найденные в тексте."""
    return [match.group(0) for match in _FORBIDDEN_RE.finditer(text or "")]


# --- разбор и проверка ответа модели -------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    event_type: str
    importance: str
    summary: str
    mechanism: str

    def as_dict(self) -> dict[str, str]:
        return {
            "event_type": self.event_type,
            "importance": self.importance,
            "summary": self.summary,
            "mechanism": self.mechanism,
        }


@dataclass(frozen=True)
class Validation:
    analysis: Analysis | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.analysis is not None


def parse_json_object(raw: str | None) -> dict | None:
    """Терпимый разбор: JSON-объект целиком, в ограде ```json ... ``` или внутри текста
    (локальная модель любит обернуть ответ словами). Не объект → None."""
    if not raw:
        return None
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidates = [text]
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        loaded = _loads_object(candidate)
        if loaded is not None:
            return loaded
    # первый сбалансированный {...} внутри текста
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def _loads_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def validate_analysis(raw: str | None, link: str) -> Validation:
    """Проверка ответа модели по схеме и перечням. `link` — ссылка на источник (непустая:
    новость без ссылки не показывается как разобранная)."""
    data = parse_json_object(raw)
    if data is None:
        return Validation(None, ("ответ не является JSON-объектом",))

    problems: list[str] = []
    event_type = _clean(data.get("event_type")).lower()
    importance = _clean(data.get("importance")).lower()
    summary = _clean(data.get("summary"))
    mechanism = _clean(data.get("mechanism"))

    if event_type not in EVENT_TYPES:
        problems.append(f"event_type должен быть одним из: {', '.join(EVENT_TYPES)}")
    if importance not in IMPORTANCE_LEVELS:
        problems.append(f"importance должна быть одной из: {', '.join(IMPORTANCE_LEVELS)}")
    for name, value, limit in (
        ("summary", summary, MAX_SUMMARY_CHARS),
        ("mechanism", mechanism, MAX_MECHANISM_CHARS),
    ):
        if not value:
            problems.append(f"поле {name} пустое")
        elif len(value) > limit:
            problems.append(f"поле {name} длиннее {limit} символов")
        else:
            forbidden = find_forbidden(value)
            if forbidden:
                shown = ", ".join(sorted({word.lower() for word in forbidden}))
                problems.append(
                    f"в поле {name} запрещённые формулировки (направление, прогноз, совет "
                    f"или оценка): {shown}"
                )
    if not link or not link.strip():
        problems.append("у новости нет ссылки на источник")

    if problems:
        return Validation(None, tuple(problems))
    return Validation(Analysis(event_type, importance, summary, mechanism), ())


ACCEPT, RETRY, FALLBACK = "accept", "retry", "fallback"


def next_step(attempt: int, validation: Validation) -> str:
    """Что делать после `attempt`-й попытки (с 1): принять, повторить или показать без разбора."""
    if validation.ok:
        return ACCEPT
    return RETRY if attempt < MAX_ATTEMPTS else FALLBACK


# --- выбор элементов сообщений и сроки ------------------------------------------------------


def sort_items(items: list[dict]) -> list[dict]:
    """Сначала важные, внутри уровня — свежие. Новость без разбора — после всех разобранных."""

    def rank(item: dict) -> int:
        analysis = item.get("analysis") or {}
        return IMPORTANCE_RANK.get(analysis.get("importance"), len(IMPORTANCE_RANK))

    by_time = sorted(items, key=lambda item: item.get("published_at") or "", reverse=True)
    return sorted(by_time, key=rank)


def is_urgent(item: dict) -> bool:
    """Срочное — только разобранная новость с высокой важностью."""
    analysis = item.get("analysis")
    return bool(analysis) and analysis.get("importance") == "high"


def messages_left(sent_today: int, max_per_day: int) -> int:
    return max(max_per_day - sent_today, 0)


def select_urgent(items: list[dict], budget: int) -> list[dict]:
    """Срочные сообщения: по одному на новость, не больше бюджета суточного лимита. Бюджет
    сохраняет одно сообщение для дайджеста, если дайджест ещё впереди (reserve в вызове)."""
    if budget <= 0:
        return []
    return [item for item in sort_items(items) if is_urgent(item)][:budget]


def select_digest(items: list[dict], max_items: int) -> list[dict]:
    """Элементы дайджеста: важные и свежие первыми, не более `max_items`; остальное остаётся
    неподтверждённым и войдёт в следующий раз."""
    return sort_items(items)[: max(max_items, 0)]


def parse_moment(value: object) -> datetime | None:
    """ISO-момент сервера → aware datetime (без пояса — UTC); непонятное → None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def next_digest_at(
    enabled_at: datetime, last_digest_at: datetime | None, interval_hours: int
) -> datetime:
    """Когда наступает срок дайджеста: период после последнего (или после включения)."""
    return (last_digest_at or enabled_at) + timedelta(hours=interval_hours)
