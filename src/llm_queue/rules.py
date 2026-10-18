"""Чистые правила очереди LLM-задач: порядок, оценка ожидания, лимиты, тексты.

Не знает ни про Telegram, ни про asyncio, ни про `config`: всё внедряется аргументами, поэтому
правила проверяются тестами без сети (tests/test_llm_queue_rules.py). Шлюз со слотами и
реестром задач — `gate.py`; команды — `commands.py`. См. design.md изменения
add-llm-request-queue (решения 3, 7, 8, 9).
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Приоритеты: чем меньше число, тем раньше слот. Внутри приоритета — по порядку постановки.
INTERACTIVE = 0
RESEARCH = 1

# Виды задач: у каждого своя статистика длительностей (ответ основного потока и цикл
# /smart_agent с инструментами различаются на порядок).
KIND_MAIN = "main"
KIND_AGENT = "agent"
KIND_COMPARE = "compare"
KIND_SMART_AGENT = "smart_agent"
KIND_DEV_ASSISTANT = "dev_assistant"
KIND_SUPPORT = "support"
KIND_RESEARCH = "research"
KIND_NEWS = "news"  # фоновый разбор новостей: один слот на новость, без показа пользователю
KINDS = (
    KIND_MAIN,
    KIND_AGENT,
    KIND_COMPARE,
    KIND_SMART_AGENT,
    KIND_DEV_ASSISTANT,
    KIND_SUPPORT,
    KIND_RESEARCH,
    KIND_NEWS,
)

# Сколько последних длительностей хранится по виду и с какого числа им верят.
DURATION_WINDOW = 20
MIN_SAMPLES = 3

# Начало вопроса в списке /queue: несколько слов, чтобы пользователь узнал вопрос.
SNIPPET_MAX_WORDS = 6
SNIPPET_MAX_CHARS = 60

CANCEL_COMMAND = "/queue_cancel"
QUEUE_COMMAND = "/queue"


@dataclass(frozen=True)
class QueueEntry:
    """Задача в очереди — только то, что нужно правилам порядка и оценки."""

    seq: int  # номер постановки (монотонный): порядок FIFO
    chat_id: int
    kind: str
    priority: int
    running: bool = False


@dataclass(frozen=True)
class Estimate:
    """Диапазон ожидания ответа в минутах; low=None — измерений мало, нижней границы нет."""

    low_minutes: int | None
    high_minutes: int


# --- Порядок и позиция ---------------------------------------------------------------------


def _wait_order_key(entry: QueueEntry) -> tuple[int, int]:
    return (entry.priority, entry.seq)


def order_waiting(entries: Iterable[QueueEntry]) -> list[QueueEntry]:
    """Ожидающие задачи в порядке выдачи слота: приоритет, затем FIFO."""
    return sorted((e for e in entries if not e.running), key=_wait_order_key)


def entries_ahead(entries: Sequence[QueueEntry], target: QueueEntry) -> list[QueueEntry]:
    """Задачи, которым слот достаётся раньше target: все выполняемые и ожидающие впереди.

    Для выполняемой задачи — пустой список. Выполняемые идут первыми (в порядке постановки),
    затем ожидающие в порядке выдачи слота.
    """
    if target.running:
        return []
    running = sorted((e for e in entries if e.running and e.seq != target.seq), key=lambda e: e.seq)
    waiting = [
        e
        for e in order_waiting(entries)
        if e.seq != target.seq and _wait_order_key(e) < _wait_order_key(target)
    ]
    return running + waiting


def count_ahead(entries: Sequence[QueueEntry], target: QueueEntry) -> int:
    """Сколько задач впереди target (выполняемые и ожидающие раньше)."""
    return len(entries_ahead(entries, target))


def queue_position(entries: Sequence[QueueEntry], target: QueueEntry) -> int:
    """Позиция в общей очереди: 0 — выполняется, 1 — следующая за выполняемыми и т. д."""
    if target.running:
        return 0
    return count_ahead(entries, target) + 1


# --- Статистика длительностей и оценка -----------------------------------------------------


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Перцентиль по методу ближайшего ранга (values непусты)."""
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


class DurationStats:
    """Скользящее окно длительностей выполнения по видам задач (только в памяти)."""

    def __init__(self, window: int = DURATION_WINDOW) -> None:
        self._window = window
        self._by_kind: dict[str, deque[float]] = {}

    def record(self, kind: str, seconds: float) -> None:
        if seconds < 0:
            return
        self._by_kind.setdefault(kind, deque(maxlen=self._window)).append(seconds)

    def count(self, kind: str) -> int:
        return len(self._by_kind.get(kind, ()))

    def median(self, kind: str) -> float | None:
        values = self._by_kind.get(kind)
        return _median(list(values)) if values else None

    def p90(self, kind: str) -> float | None:
        values = self._by_kind.get(kind)
        return _percentile(list(values), 0.9) if values else None

    def has_enough(self, kind: str) -> bool:
        return self.count(kind) >= MIN_SAMPLES


def _round_up_minutes(seconds: float) -> int:
    return max(1, math.ceil(seconds / 60))


def estimate_wait(
    stats: DurationStats,
    ahead_kinds: Sequence[str],
    own_kind: str,
    concurrency: int,
    timeout_seconds: float,
) -> Estimate:
    """Диапазон ожидания ОТВЕТА для задачи с ahead_kinds впереди и параллелизмом K.

    До начала своей задачи надо дождаться ceil(n / K) «волн» задач впереди (берётся каждая
    K-я по порядку — по виду именно этих задач), затем выполнить свою. Нижняя граница —
    сумма медиан, верхняя — сумма 90-х процентилей. Если по какому-то виду меньше
    MIN_SAMPLES измерений, нижней границы нет, а верхняя для этого вида — предельное время
    обращения. Сумма округляется вверх до минуты.
    """
    k = max(1, concurrency)
    waves = list(ahead_kinds)[::k]
    low_total = 0.0
    high_total = 0.0
    low_known = True
    for kind in [*waves, own_kind]:
        if stats.has_enough(kind):
            low_total += stats.median(kind) or 0.0
            high_total += stats.p90(kind) or 0.0
        else:
            low_known = False
            high_total += timeout_seconds
    return Estimate(
        low_minutes=_round_up_minutes(low_total) if low_known else None,
        high_minutes=_round_up_minutes(high_total),
    )


# --- Лимиты --------------------------------------------------------------------------------


def chat_limit_reached(chat_jobs: int, limit: int) -> bool:
    """У чата уже есть limit ожидающих/выполняемых вопросов — новый не принимается."""
    return chat_jobs >= limit


def queue_full(waiting_interactive: int, queue_max: int) -> bool:
    """Ожидающих вопросов пользователей уже queue_max — отказ «бот занят»."""
    return waiting_interactive >= queue_max


def ttl_expired(waited_seconds: float, ttl_seconds: float) -> bool:
    """Вопрос ждал слот дольше срока."""
    return waited_seconds > ttl_seconds


# --- Кооперативная отмена вопроса агента ---------------------------------------------------


class AskCancelled(Exception):  # noqa: N818 — не ошибка, а штатный исход отмены пользователем
    """Вопрос агента снят пользователем до записи обмена: состояние не менялось."""


def raise_if_cancelled(should_cancel) -> None:
    """Бросает `AskCancelled`, если вопрос снят. `should_cancel=None` — отмены нет.

    Вызывается агентами между LLM-вызовами и ПЕРЕД записью обмена в память/историю:
    прервать идущий HTTP-вызов в потоке нельзя, но снятый вопрос не должен записаться.
    """
    if should_cancel is not None and should_cancel():
        raise AskCancelled()


# --- Аргумент /queue_cancel ----------------------------------------------------------------

_INT_RE = re.compile(r"^-?[0-9]+$")


@dataclass(frozen=True)
class CancelArg:
    """Разобранный аргумент отмены: все вопросы, номер или ошибка формы."""

    all_jobs: bool = False
    number: int | None = None
    error: bool = False


def parse_cancel_arg(args: Sequence[str]) -> CancelArg:
    """Пусто — все вопросы; одно целое — номер; остальное — ошибка.

    Существование номера проверяет `select_by_number`: `0` и `-1` — целые, но вопроса с
    таким номером нет.
    """
    tokens = [a for a in (arg.strip() for arg in args) if a]
    if not tokens:
        return CancelArg(all_jobs=True)
    if len(tokens) == 1 and _INT_RE.match(tokens[0]):
        return CancelArg(number=int(tokens[0]))
    return CancelArg(error=True)


def select_by_number(jobs: Sequence, number: int):
    """Вопрос по номеру из `/queue` (нумерация с 1 в порядке отправки) или None."""
    if 1 <= number <= len(jobs):
        return jobs[number - 1]
    return None


# --- Тексты --------------------------------------------------------------------------------


def snippet(
    text: str, max_words: int = SNIPPET_MAX_WORDS, max_chars: int = SNIPPET_MAX_CHARS
) -> str:
    """Начало вопроса для списка: несколько слов, не длиннее max_chars."""
    words = text.split()
    if not words:
        return ""
    cut = words[:max_words]
    result = " ".join(cut)
    truncated = len(words) > max_words
    if len(result) > max_chars:
        result = result[:max_chars].rstrip()
        truncated = True
    return result + "…" if truncated else result


def format_estimate(estimate: Estimate) -> str:
    """«примерно X–Y мин» / «примерно до Y мин» / «примерно X мин»."""
    low, high = estimate.low_minutes, estimate.high_minutes
    if low is None:
        return f"примерно до {high} мин"
    if low >= high:
        return f"примерно {high} мин"
    return f"примерно {low}–{high} мин"


_ESTIMATE_NOTE = "Это приблизительная оценка, а не обещание: время зависит от загрузки модели."


def text_accepted(position: int, estimate: Estimate) -> str:
    return (
        f"Вопрос принят в очередь. Позиция: {position}. "
        f"Ответ придёт {format_estimate(estimate)}.\n"
        f"{_ESTIMATE_NOTE}\n"
        f"Посмотреть очередь: {QUEUE_COMMAND}. Снять вопрос: {CANCEL_COMMAND}."
    )


def text_chat_limit(position: int, estimate: Estimate | None) -> str:
    where = "выполняется" if position == 0 else f"позиция {position}"
    when = f" Ответ придёт {format_estimate(estimate)}." if estimate is not None else ""
    return (
        f"Предыдущий вопрос ещё обрабатывается ({where}).{when} "
        f"Дождитесь ответа или снимите вопрос командой {CANCEL_COMMAND}."
    )


def text_busy() -> str:
    return "Бот сейчас занят: очередь заполнена. Повторите вопрос чуть позже."


def text_expired() -> str:
    return (
        "Вопрос слишком долго ждал в очереди и снят, ответа на него не будет. "
        "Отправьте его ещё раз."
    )


@dataclass(frozen=True)
class QueueItemView:
    """Пункт списка /queue: номер, позиция (0 — выполняется), оценка и начало вопроса."""

    number: int
    position: int
    estimate: Estimate | None
    text: str


def format_queue_list(items: Sequence[QueueItemView]) -> str:
    """Нумерованный список вопросов пользователя; пустой — «очередь пуста»."""
    if not items:
        return "Ваших вопросов в очереди нет."
    lines = ["Ваши вопросы в очереди:"]
    for item in items:
        if item.position == 0:
            status = "выполняется"
        else:
            status = f"позиция {item.position}"
            if item.estimate is not None:
                status += f", ответ {format_estimate(item.estimate)}"
        lines.append(f"{item.number}. {snippet(item.text)} — {status}")
    lines.append(_ESTIMATE_NOTE)
    lines.append(f"Снять один: {CANCEL_COMMAND} <номер>, все: {CANCEL_COMMAND}.")
    return "\n".join(lines)


def text_cancel_all(count: int) -> str:
    if count <= 0:
        return text_nothing_to_cancel()
    return f"Снято вопросов: {count}. Ответы на них не придут."


def text_cancel_one(number: int, running: bool) -> str:
    if running:
        return (
            f"Вопрос {number} снят: выполнение прервано, ответ не придёт. "
            "Следующий вопрос начнётся, когда модель закончит текущий вызов."
        )
    return f"Вопрос {number} снят с очереди, ответ не придёт."


def text_nothing_to_cancel() -> str:
    return f"Отменять нечего: ваших вопросов в очереди нет. Список: {QUEUE_COMMAND}."


def text_bad_number() -> str:
    return (
        f"Не нашёл такого вопроса. Номера смотрите в {QUEUE_COMMAND}, "
        f"затем {CANCEL_COMMAND} <номер> (например, {CANCEL_COMMAND} 2) "
        f"или {CANCEL_COMMAND} без номера, чтобы снять все."
    )


def text_already_saved() -> str:
    return (
        "Вопрос снят, но обмен уже был записан в память агента: ответ не отправляется. "
        "Посмотреть и очистить память можно командами агента."
    )
