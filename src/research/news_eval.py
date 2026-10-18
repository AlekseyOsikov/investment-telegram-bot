"""Чистые правила оценки качества разбора новостей (скрипт scripts/news_eval.py, спека news-digest,
требование «Оценка качества разбора»; design.md изменения add-news-digest, решение 8).

Отделено от скрипта по принципу проекта (как research/dialog_eval.py): здесь ПРАВИЛА — формат
набора, прогон одной новости с теми же сообщениями, проверками и повтором, что в боте, итоги и
текст отчёта; обращение к модели и диск — в скрипте. Модуль не импортирует config, openai и
Telegram: вызов модели внедряется (`complete`), поэтому всё проверяется тестами
(tests/test_news_eval.py) без сети.

Набор — публичные новости из каталога данных (не коммитятся). В тестах названия вымышленные.
Проверки — ровно `news_digest.rules.validate_analysis`: допустимость типа и важности, длины,
отсутствие запрещённых формулировок, наличие ссылки; плюс доля совпадения типа события с
необязательной разметкой. Результат — отчёт оператору; настройки скрипт не применяет.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from news_digest import prompts, rules

Complete = Callable[[list[dict]], str]

NO_DATA_TEXT = "Оценивать нечего: набор новостей не найден (каталог data/news_eval/ пуст)."


class DatasetError(ValueError):
    """Файл набора не соответствует формату (сообщение — по-русски, для оператора)."""


@dataclass(frozen=True)
class EvalItem:
    title: str
    excerpt: str = ""
    source: str = "набор"
    link: str = ""
    published_at: str = ""
    tickers: tuple[str, ...] = ()
    expected_type: str | None = None  # необязательная разметка


@dataclass(frozen=True)
class Dataset:
    name: str
    items: tuple[EvalItem, ...]


@dataclass
class ItemResult:
    item: EvalItem
    analysis: dict | None = None
    attempts: int = 0
    problems: list[str] = field(default_factory=list)  # нарушения последней попытки
    error: str = ""  # модель не ответила

    @property
    def ok(self) -> bool:
        return self.analysis is not None

    @property
    def type_matches(self) -> bool | None:
        """None — разметки нет или разбора нет; иначе совпал ли тип события."""
        if self.item.expected_type is None or self.analysis is None:
            return None
        return self.analysis["event_type"] == self.item.expected_type


def parse_dataset(data: object, default_name: str = "набор") -> Dataset:
    """Набор из JSON: `{"name": "...", "items": [{"title": "...", "excerpt": "...", "source":
    "...", "link": "...", "tickers": ["ALFA"], "expected_type": "dividends"}]}`."""
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise DatasetError("Набор должен быть объектом с полем items — списком новостей.")
    items: list[EvalItem] = []
    for index, raw in enumerate(data["items"], start=1):
        if not isinstance(raw, dict) or not str(raw.get("title") or "").strip():
            raise DatasetError(f"Новость №{index}: нужен непустой title.")
        expected = raw.get("expected_type")
        if expected is not None and expected not in rules.EVENT_TYPES:
            raise DatasetError(
                f"Новость №{index}: expected_type {expected!r} не из перечня "
                f"({', '.join(rules.EVENT_TYPES)})."
            )
        tickers = raw.get("tickers") or []
        if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
            raise DatasetError(f"Новость №{index}: tickers должен быть списком строк.")
        items.append(
            EvalItem(
                title=str(raw["title"]).strip(),
                excerpt=str(raw.get("excerpt") or ""),
                source=str(raw.get("source") or "набор"),
                link=str(raw.get("link") or ""),
                published_at=str(raw.get("published_at") or ""),
                tickers=tuple(t.strip().upper() for t in tickers),
                expected_type=expected,
            )
        )
    if not items:
        raise DatasetError("В наборе нет новостей.")
    return Dataset(str(data.get("name") or default_name), tuple(items))


def to_news(item: EvalItem, index: int) -> dict:
    """Новость в том виде, в каком её отдаёт бот модели (запись сервера новостей)."""
    return {
        "id": f"eval-{index}",
        "source": item.source,
        "link": item.link,
        "title": item.title,
        "excerpt": item.excerpt,
        "published_at": item.published_at,
        "tickers": [{"ticker": t, "matched": t, "kind": "ticker"} for t in item.tickers],
    }


def evaluate_item(
    item: EvalItem, index: int, complete: Complete, base_system_prompt: str | None
) -> ItemResult:
    """Одна новость: те же сообщения, проверка и один повтор, что в боте; исход — разбор либо
    нарушения последней попытки либо ошибка модели."""
    news = to_news(item, index)
    result = ItemResult(item)
    problems: tuple[str, ...] = ()
    for attempt in range(1, rules.MAX_ATTEMPTS + 1):
        result.attempts = attempt
        messages = prompts.build_analysis_messages(news, base_system_prompt, problems)
        try:
            raw = complete(messages)
        except Exception as error:  # noqa: BLE001 — сбой модели фиксируется в отчёте
            result.error = f"{type(error).__name__}: {error}"
            return result
        validation = rules.validate_analysis(raw, item.link)
        step = rules.next_step(attempt, validation)
        if step == rules.ACCEPT:
            result.analysis = validation.analysis.as_dict()
            result.problems = []
            return result
        result.problems = list(validation.problems)
        problems = validation.problems
        if step == rules.FALLBACK:
            return result
    return result


@dataclass(frozen=True)
class Summary:
    total: int
    accepted: int
    first_try: int
    fallback: int
    errors: int
    labeled: int
    type_matched: int
    violations: dict[str, int]

    @property
    def type_share(self) -> float | None:
        return self.type_matched / self.labeled if self.labeled else None


def _problem_key(problem: str) -> str:
    """Группа нарушения для сводки: тип проблемы без подробностей."""
    if problem.startswith("в поле"):
        return "запрещённые формулировки"
    if "JSON" in problem:
        return "ответ не JSON"
    if problem.startswith("event_type"):
        return "тип события не из перечня"
    if problem.startswith("importance"):
        return "важность не из перечня"
    if "длиннее" in problem:
        return "слишком длинное поле"
    if "пустое" in problem:
        return "пустое поле"
    if "ссылк" in problem:
        return "нет ссылки"
    return problem


def summarize(results: list[ItemResult]) -> Summary:
    violations: Counter[str] = Counter()
    for result in results:
        for problem in result.problems if not result.ok else []:
            violations[_problem_key(problem)] += 1
    labeled = [r for r in results if r.type_matches is not None]
    return Summary(
        total=len(results),
        accepted=sum(1 for r in results if r.ok),
        first_try=sum(1 for r in results if r.ok and r.attempts == 1),
        fallback=sum(1 for r in results if not r.ok and not r.error),
        errors=sum(1 for r in results if r.error),
        labeled=len(labeled),
        type_matched=sum(1 for r in labeled if r.type_matches),
        violations=dict(violations),
    )


def render_report(dataset: Dataset, results: list[ItemResult], summary: Summary, model: str) -> str:
    """Текст отчёта оператору (с текстами новостей — хранить в data/, не в репозитории)."""
    share = summary.type_share
    lines = [
        f"# Оценка разбора новостей: {dataset.name}",
        f"Модель: {model}",
        "",
        f"Новостей: {summary.total}. Принято кодом: {summary.accepted} "
        f"(с первой попытки: {summary.first_try}). Без разбора после проверки: {summary.fallback}. "
        f"Ошибок модели: {summary.errors}.",
    ]
    if share is not None:
        lines.append(
            f"Совпадение типа события с разметкой: {summary.type_matched} из {summary.labeled} "
            f"({share * 100:.0f}%)."
        )
    else:
        lines.append("Разметки типа нет — доля совпадения не считается.")
    if summary.violations:
        lines += ["", "Нарушения в отвергнутых ответах:"]
        lines += [f"- {name}: {count}" for name, count in sorted(summary.violations.items())]
    lines += ["", "## По новостям"]
    for index, result in enumerate(results, start=1):
        head = f"{index}. {result.item.title}"
        if result.ok:
            matched = result.type_matches
            mark = (
                ""
                if matched is None
                else (
                    " — тип совпал"
                    if matched
                    else f" — тип НЕ совпал (ожидался {result.item.expected_type})"
                )
            )
            lines.append(
                f"{head}\n   принято (попыток: {result.attempts}), "
                f"{result.analysis['event_type']}/{result.analysis['importance']}{mark}"
            )
        elif result.error:
            lines.append(f"{head}\n   ошибка модели: {result.error}")
        else:
            lines.append(f"{head}\n   отвергнуто: {'; '.join(result.problems)}")
    lines += [
        "",
        "Отчёт ничего не меняет: модель, промпт и настройки остаются решением оператора.",
    ]
    return "\n".join(lines)
