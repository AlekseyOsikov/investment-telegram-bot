"""Чистые правила сравнения РЕЖИМОВ ПОИСКА слоя rag (фильтр, переписывание вопроса) в команде
/research_rag_compare (design.md изменения add-rag-rerank-and-rewrite, решение 6).

Дополняет research/rag_compare_eval.py: там — сравнение ответов «без RAG / с RAG», здесь — режимы
поиска, их выбор оператором, метрики попадания в ожидаемый документ и тексты отчёта. Модуль не
импортирует ни openai, ни faiss, ни config, ни Telegram — все ветвления проверяются тестами
(tests/test_rag_modes_eval.py) без сети. Обращения к SmartAgent, rewrite-модели, диску и
Telegram — в research/rag_compare.py.

Режимы поиска (RETRIEVAL_MODES):
  baseline        — как до изменения: топ-K по близости и порог, без отбора и переписывания;
  filter          — второй этап отбора (RAG_CANDIDATES кандидатов → шаги отбора → топ-K);
  rewrite         — переписывание вопроса, поиск как в baseline;
  rewrite_filter  — переписывание и отбор (рабочий режим при настроенном переписывании);
  rewrite_context — переписывание с предыдущими вопросами диалога (поле `history` вопроса в
                    наборе; design.md изменения add-rag-rewrite-dialog-context);
  concat          — без моделей: эмбеддинг по «последний предыдущий вопрос + текущий».
Настройки каждого режима, кроме отличающихся, берутся из рабочей конфигурации.

Отчёт режима «только поиск» — обычный dict (kind="retrieval"), его пишет и читает rag_compare.py
как JSON; агрегат считается из результатов на лету.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import rag_compare_eval as ev

MODE_BASELINE = "baseline"
MODE_FILTER = "filter"
MODE_REWRITE = "rewrite"
MODE_REWRITE_FILTER = "rewrite_filter"
MODE_REWRITE_CONTEXT = "rewrite_context"
MODE_CONCAT = "concat"

RETRIEVAL_MODES = (
    MODE_BASELINE,
    MODE_FILTER,
    MODE_REWRITE,
    MODE_REWRITE_FILTER,
    MODE_REWRITE_CONTEXT,
    MODE_CONCAT,
)
MODE_TITLES = {
    MODE_BASELINE: "как до изменения",
    MODE_FILTER: "с отбором",
    MODE_REWRITE: "с переписыванием",
    MODE_REWRITE_FILTER: "с переписыванием и отбором",
    MODE_REWRITE_CONTEXT: "с контекстом диалога",
    MODE_CONCAT: "склейка",
}
_REWRITE_MODES = (MODE_REWRITE, MODE_REWRITE_FILTER, MODE_REWRITE_CONTEXT)
_FILTER_MODES = (MODE_FILTER, MODE_REWRITE_FILTER)
# Режимы, которым нужна история диалога вопроса (поле `history` набора): на самостоятельном
# вопросе они совпадают с «как до изменения» («склейка») и «с переписыванием» («с контекстом»).
_HISTORY_MODES = (MODE_REWRITE_CONTEXT, MODE_CONCAT)

LEVEL_SEARCH = "search"  # только поиск: ни ответов модели, ни оценщика
LEVEL_ANSWERS = "answers"  # ответы и оценка по фактам (прежний прогон «без RAG / с RAG»)
LEVELS = (LEVEL_SEARCH, LEVEL_ANSWERS)

REPORT_KIND_RETRIEVAL = "retrieval"
REWRITE_NOT_CONFIGURED = "переписывание не настроено (REWRITE_PROVIDER)"
HISTORY_MODE_ANSWERS_REASON = (
    "требует вопросов с историей диалога; такие вопросы проверяются на уровне search"
)

# Эталонное число предыдущих вопросов для режима «с контекстом диалога», если у оператора
# история в переписывании выключена (REWRITE_HISTORY_QUESTIONS=0): иначе режим не проверял бы
# того, ради чего он нужен. Указывается в отчёте (design.md add-rag-rewrite-dialog-context).
REFERENCE_HISTORY_QUESTIONS = 3


# Эталонные значения шагов отбора для режимов «с отбором»: если у оператора все шаги выключены
# (так по умолчанию), режим сравнивался бы сам с собой. Значения — из первого сравнения режимов
# (design.md изменения add-rag-rerank-and-rewrite); указываются в отчёте.
REFERENCE_FILTER = {
    "candidates": 10,
    "relative_margin": 0.08,
    "min_chunk_chars": 120,
    "max_per_doc": 2,
}


class ModeSettingError(ValueError):
    """Недопустимая настройка режимов или уровня сравнения — прогон не начинается."""


def filter_steps_neutral(
    top_k: int, relative_margin: float, min_chunk_chars: int, max_per_doc: int
) -> bool:
    """Все шаги отбора (кроме порога) выключены нейтральными значениями."""
    return relative_margin >= 1.0 and min_chunk_chars <= 0 and max_per_doc >= top_k


def filter_settings(
    candidates: int, top_k: int, relative_margin: float, min_chunk_chars: int, max_per_doc: int
) -> tuple[dict, bool]:
    """Настройки отбора для режимов «с отбором»: рабочие, а если все шаги выключены — эталонные
    REFERENCE_FILTER (кандидатов не меньше рабочего числа). Возвращает (настройки, взяты ли
    эталонные)."""
    if filter_steps_neutral(top_k, relative_margin, min_chunk_chars, max_per_doc):
        settings = dict(REFERENCE_FILTER)
        settings["candidates"] = max(candidates, REFERENCE_FILTER["candidates"])
        return settings, True
    return (
        {
            "candidates": candidates,
            "relative_margin": relative_margin,
            "min_chunk_chars": min_chunk_chars,
            "max_per_doc": max_per_doc,
        },
        False,
    )


def history_questions_setting(configured: int) -> tuple[int, bool]:
    """Сколько предыдущих вопросов передавать режиму «с контекстом диалога»: рабочее значение, а
    если оно нулевое — эталонное REFERENCE_HISTORY_QUESTIONS. Возвращает (число, взято ли
    эталонное)."""
    if configured > 0:
        return configured, False
    return REFERENCE_HISTORY_QUESTIONS, True


def concat_text(history: list[str] | tuple[str, ...], question: str) -> str:
    """Текст режима «склейка»: последний предыдущий вопрос диалога и текущий через пробел; у
    вопроса без истории — сам вопрос (режим совпадает с «как до изменения»)."""
    if not history:
        return question
    return f"{history[-1]} {question}"


def is_multi_turn(result: QuestionRetrieval) -> bool:
    """Многоходовый ли вопрос: в наборе у него задана история диалога."""
    return bool(result.question.history)


def split_by_history(
    results: list[QuestionRetrieval],
) -> tuple[list[QuestionRetrieval], list[QuestionRetrieval]]:
    """(многоходовые, самостоятельные) вопросы — метрики режимов считаются по группам
    отдельно, чтобы выигрыш контекста диалога не терялся среди вопросов, у которых история
    ничего не меняет."""
    multi = [r for r in results if is_multi_turn(r)]
    single = [r for r in results if not is_multi_turn(r)]
    return multi, single


def uses_history(mode_key: str) -> bool:
    return mode_key in _HISTORY_MODES


def uses_rewrite(mode_key: str) -> bool:
    return mode_key in _REWRITE_MODES


def uses_filter(mode_key: str) -> bool:
    return mode_key in _FILTER_MODES


def parse_modes(raw: str) -> list[str]:
    """Режимы из настройки оператора (RAG_COMPARE_MODES): список через запятую, регистр и
    пробелы не важны, порядок — порядок RETRIEVAL_MODES, повторы убираются. Пустая строка —
    пустой список («явного выбора нет»); неизвестное имя — ModeSettingError."""
    names = [part.strip().lower() for part in (raw or "").split(",") if part.strip()]
    unknown = [name for name in names if name not in RETRIEVAL_MODES]
    if unknown:
        raise ModeSettingError(
            f"неизвестные режимы в RAG_COMPARE_MODES: {', '.join(unknown)}. "
            f"Допустимо: {', '.join(RETRIEVAL_MODES)}."
        )
    return [mode for mode in RETRIEVAL_MODES if mode in names]


def parse_level(raw: str) -> str:
    level = (raw or "").strip().lower()
    if level not in LEVELS:
        raise ModeSettingError(
            f"недопустимое RAG_COMPARE_LEVEL={raw!r}. Допустимо: {', '.join(LEVELS)}."
        )
    return level


def resolve_modes(
    selected: list[str], level: str, rewrite_configured: bool
) -> tuple[list[str], dict[str, str]]:
    """(активные режимы, недоступные режимы с причиной). Пустой выбор на уровне поиска — все
    режимы, на уровне ответов — ни одного дополнительного (остаётся прежнее сравнение «без
    RAG / с RAG»). Режимы с переписыванием при ненастроенном переписывании недоступны: их
    нельзя молча подменить другим режимом."""
    if selected:
        chosen = list(selected)
    else:
        chosen = list(RETRIEVAL_MODES) if level == LEVEL_SEARCH else []
    active: list[str] = []
    unavailable: dict[str, str] = {}
    for mode in chosen:
        if level == LEVEL_ANSWERS and uses_history(mode):
            unavailable[mode] = HISTORY_MODE_ANSWERS_REASON
        elif uses_rewrite(mode) and not rewrite_configured:
            unavailable[mode] = REWRITE_NOT_CONFIGURED
        else:
            active.append(mode)
    return active, unavailable


# --------------------------------------------------------------------------- #
# Результаты поиска по режимам
# --------------------------------------------------------------------------- #


@dataclass
class RetrievalResult:
    """Итог одного режима по одному вопросу: отобранные чанки (по убыванию близости),
    сколько кандидатов найдено, время поиска; error — причина сбоя поиска; history_used — сколько
    предыдущих вопросов диалога реально ушло в переписывание (режим «с контекстом диалога»)."""

    chunks: list[dict] = field(default_factory=list)  # {"title","chunk_index","score"}
    candidates: int = 0
    search_text: str | None = None  # переписанный запрос (None — искали по вопросу как есть)
    seconds: float = 0.0
    error: str | None = None
    history_used: int = 0

    def to_dict(self) -> dict:
        return {
            "chunks": [dict(chunk) for chunk in self.chunks],
            "candidates": self.candidates,
            "search_text": self.search_text,
            "seconds": self.seconds,
            "error": self.error,
            "history_used": self.history_used,
        }

    @staticmethod
    def from_dict(data: dict) -> RetrievalResult:
        return RetrievalResult(
            chunks=[dict(chunk) for chunk in data.get("chunks", [])],
            candidates=int(data.get("candidates", 0)),
            search_text=data.get("search_text"),
            seconds=float(data.get("seconds", 0.0)),
            error=data.get("error"),
            history_used=int(data.get("history_used", 0)),
        )

    @property
    def titles(self) -> list[str]:
        return [chunk.get("title", "") for chunk in self.chunks]


@dataclass
class QuestionRetrieval:
    """Результаты всех режимов по одному вопросу плюс итог переписывания (один раз на вопрос,
    результат переиспользуется между режимами с переписыванием). Переписывание с историей
    диалога (режим «с контекстом диалога») хранится отдельно: `context_rewrite_*` заполняется
    только у вопроса с историей."""

    question: ev.Question
    modes: dict[str, RetrievalResult] = field(default_factory=dict)
    rewrite_query: str | None = None
    rewrite_failure: str | None = None
    rewrite_seconds: float = 0.0
    context_rewrite_query: str | None = None
    context_rewrite_failure: str | None = None
    context_rewrite_seconds: float = 0.0
    missing_sources: list[str] = field(default_factory=list)  # ожидаемые, которых нет в индексе

    @property
    def source_missing(self) -> bool:
        return bool(self.missing_sources)

    def to_dict(self) -> dict:
        q = self.question
        return {
            "number": q.number,
            "question": q.question,
            "kind": q.kind,
            "facts": list(q.facts),
            "sources": list(q.sources),
            "history": list(q.history),
            "missing_sources": list(self.missing_sources),
            "rewrite": {
                "query": self.rewrite_query,
                "failure": self.rewrite_failure,
                "seconds": self.rewrite_seconds,
            },
            "context_rewrite": {
                "query": self.context_rewrite_query,
                "failure": self.context_rewrite_failure,
                "seconds": self.context_rewrite_seconds,
            },
            "modes": {key: result.to_dict() for key, result in self.modes.items()},
        }

    @staticmethod
    def from_dict(data: dict) -> QuestionRetrieval:
        question = ev.Question(
            number=int(data["number"]),
            question=data["question"],
            kind=data["kind"],
            facts=tuple(data.get("facts", ())),
            sources=tuple(data["sources"]),
            history=tuple(data.get("history") or ()),
        )
        rewrite = data.get("rewrite") or {}
        context_rewrite = data.get("context_rewrite") or {}
        return QuestionRetrieval(
            question=question,
            modes={
                key: RetrievalResult.from_dict(value)
                for key, value in (data.get("modes") or {}).items()
            },
            rewrite_query=rewrite.get("query"),
            rewrite_failure=rewrite.get("failure"),
            rewrite_seconds=float(rewrite.get("seconds", 0.0)),
            context_rewrite_query=context_rewrite.get("query"),
            context_rewrite_failure=context_rewrite.get("failure"),
            context_rewrite_seconds=float(context_rewrite.get("seconds", 0.0)),
            missing_sources=list(data.get("missing_sources", [])),
        )


def first_hit_position(expected: tuple[str, ...] | list[str], titles: list[str]) -> int | None:
    """Позиция (с единицы) первого чанка из ожидаемого документа; None — не попал. Сопоставление
    по префиксу заголовка без учёта регистра (как ev.source_hit)."""
    for position, title in enumerate(titles, start=1):
        if ev.source_hit(expected, [title]):
            return position
    return None


def irrelevant_count(expected: tuple[str, ...] | list[str], titles: list[str]) -> int:
    """Сколько чанков выдачи — не из ожидаемых документов вопроса."""
    return sum(1 for title in titles if not ev.source_hit(expected, [title]))


def aggregate_retrieval(results: list[QuestionRetrieval], mode_keys: list[str]) -> dict[str, dict]:
    """Метрики по каждому режиму. В расчёт идут вопросы, у которых ожидаемые документы есть в
    индексе (иначе «не попал» — вина индексации, а не поиска) и поиск в этом режиме не упал.
    hit_rate — доля вопросов с попаданием; mean_position — средняя позиция первого попадания
    (по попавшим); mrr — среднее 1/позиция (промах — 0); irrelevant_share — доля чанков не из
    ожидаемых документов среди всех отобранных; empty — вопросов без единого чанка."""
    included = [r for r in results if not r.source_missing]
    aggregates: dict[str, dict] = {}
    for key in mode_keys:
        rows = [(r.question, r.modes[key]) for r in included if key in r.modes]
        ok_rows = [(q, m) for q, m in rows if m.error is None]
        positions: list[int] = []
        reciprocal = 0.0
        chunk_total = irrelevant_total = empty = 0
        for question, mode in ok_rows:
            titles = mode.titles
            position = first_hit_position(question.sources, titles)
            if position is not None:
                positions.append(position)
                reciprocal += 1.0 / position
            chunk_total += len(titles)
            irrelevant_total += irrelevant_count(question.sources, titles)
            if not titles:
                empty += 1
        count = len(ok_rows)
        aggregates[key] = {
            "questions": count,
            "failed": len(rows) - count,
            "hits": len(positions),
            "hit_rate": len(positions) / count if count else None,
            "mean_position": sum(positions) / len(positions) if positions else None,
            "mrr": reciprocal / count if count else None,
            "irrelevant_share": irrelevant_total / chunk_total if chunk_total else None,
            "mean_chunks": chunk_total / count if count else None,
            "empty": empty,
            "mean_seconds": sum(m.seconds for _, m in ok_rows) / count if count else None,
        }
    return aggregates


def question_failed(result: QuestionRetrieval) -> bool:
    """Провалился ли вопрос: поиск не удался во ВСЕХ режимах (сервер эмбеддингов недоступен,
    повреждённый индекс). Сбой переписывания вопросом не проваливает: режимы продолжают поиск
    по исходному тексту."""
    return bool(result.modes) and all(mode.error for mode in result.modes.values())


def should_stop(results: list[QuestionRetrieval], limit: int) -> bool:
    """Пора ли остановить прогон: подряд (с конца) провалилось не меньше `limit` вопросов."""
    if limit <= 0:
        return False
    streak = 0
    for result in reversed(results):
        if not question_failed(result):
            break
        streak += 1
    return streak >= limit


# --------------------------------------------------------------------------- #
# Отчёт и тексты
# --------------------------------------------------------------------------- #


def build_retrieval_report(
    *,
    started_at: str,
    finished_at: str,
    settings: dict,
    results: list[QuestionRetrieval],
    planned: int,
    modes: list[str],
    unavailable: dict[str, str],
    index_missing: list[str] | None = None,
    status: str = ev.STATUS_COMPLETED,
    stop_reason: str | None = None,
) -> dict:
    return {
        "kind": REPORT_KIND_RETRIEVAL,
        "status": status,
        "stop_reason": stop_reason,
        "planned": planned,
        "started_at": started_at,
        "finished_at": finished_at,
        "settings": settings,
        "modes": list(modes),
        "unavailable_modes": dict(unavailable),
        "index_missing": sorted(index_missing or []),
        "questions": [r.to_dict() for r in results],
    }


def is_retrieval_report(report: dict) -> bool:
    return report.get("kind") == REPORT_KIND_RETRIEVAL


def _pct(value: float | None) -> str:
    return "н/д" if value is None else f"{value * 100:.0f}%"


def _number(value: float | None, fmt: str) -> str:
    return "н/д" if value is None else format(value, fmt)


def _delta_pp(value: float | None, base: float | None) -> str:
    if value is None or base is None:
        return ""
    diff = (value - base) * 100
    return f", {diff:+.0f} п.п. к базовому" if abs(diff) >= 0.5 else ", как базовый"


def _settings_line(settings: dict) -> str:
    parts = []
    for key, label in (
        ("strategy", "стратегия"),
        ("top_k", "топ-K"),
        ("candidates", "кандидатов"),
        ("min_score", "порог"),
        ("relative_margin", "отн. порог"),
        ("min_chunk_chars", "мин. длина чанка"),
        ("max_per_doc", "чанков на документ"),
        ("embeddings_model", "эмбеддинги"),
        ("rewrite_provider", "переписывание"),
        ("rewrite_model", "модель переписывания"),
        ("rewrite_search_mode", "поиск по переписанному"),
    ):
        if settings.get(key) not in (None, ""):
            parts.append(f"{label}: {settings[key]}")
    if settings.get("history_questions") is not None:
        origin = "эталонное" if settings.get("history_reference") else "рабочее"
        parts.append(
            f"прошлых вопросов в режиме «с контекстом диалога» ({origin}): "
            f"{settings['history_questions']}"
        )
    steps = settings.get("filter_steps")
    if steps:
        reference = settings.get("filter_reference")
        origin = "эталонные, у оператора шаги выключены" if reference else "рабочие"
        parts.append(
            f"отбор в режимах «с отбором» ({origin}): кандидатов {steps.get('candidates')}, "
            f"отн. порог {steps.get('relative_margin')}, "
            f"мин. длина {steps.get('min_chunk_chars')}, на документ {steps.get('max_per_doc')}"
        )
    return "Настройки: " + ", ".join(parts) + "." if parts else ""


def _mode_blocks(aggregates: dict[str, dict], mode_keys: list[str]) -> list[str]:
    """Блоки метрик по режимам; разница с «как до изменения» — внутри тех же вопросов."""
    lines: list[str] = []
    base = aggregates.get(MODE_BASELINE)
    for key in mode_keys:
        agg = aggregates[key]
        lines.append(f"\n• {MODE_TITLES[key]}")
        delta = (
            _delta_pp(agg["hit_rate"], base["hit_rate"]) if base and key != MODE_BASELINE else ""
        )
        lines.append(
            f"  попал в ожидаемый документ: {agg['hits']} из {agg['questions']} "
            f"({_pct(agg['hit_rate'])}{delta})"
        )
        lines.append(
            f"  позиция первого попадания: {_number(agg['mean_position'], '.1f')}, "
            f"MRR: {_number(agg['mrr'], '.2f')}"
        )
        lines.append(
            f"  чанков в выдаче: {_number(agg['mean_chunks'], '.1f')}, не из ожидаемых "
            f"документов: {_pct(agg['irrelevant_share'])}, без материалов: {agg['empty']}"
        )
        lines.append(f"  время поиска: {_number(agg['mean_seconds'], '.2f')} с")
        if agg["failed"]:
            lines.append(f"  ⚠️ поиск не удался на {agg['failed']} вопросах")
    return lines


def format_retrieval_summary(report: dict) -> str:
    results = [QuestionRetrieval.from_dict(q) for q in report.get("questions", [])]
    mode_keys = [m for m in report.get("modes", []) if m in RETRIEVAL_MODES]
    aggregates = aggregate_retrieval(results, mode_keys)
    lines = ["📊 Сравнение режимов поиска (без ответов модели)"]
    if report.get("status") == ev.STATUS_STOPPED:
        planned = report.get("planned", len(results))
        reason = report.get("stop_reason") or "причина не указана"
        lines.append(
            f"⛔ Прогон остановлен: {reason}. Обработано {len(results)} из {planned} вопросов; "
            "метрики ниже — только по обработанным."
        )
    if report.get("finished_at"):
        verb = "остановлен" if report.get("status") == ev.STATUS_STOPPED else "завершён"
        lines.append(f"Прогон {verb}: {report['finished_at']} (UTC).")
    settings = _settings_line(report.get("settings", {}))
    if settings:
        lines.append(settings)

    excluded = [r.question.number for r in results if r.source_missing]
    included = len(results) - len(excluded)
    lines.append(f"\nВопросов в расчёте: {included} из {len(results)}.")

    unavailable = report.get("unavailable_modes") or {}
    for key in RETRIEVAL_MODES:
        if key in unavailable:
            lines.append(f"⚠️ Режим «{MODE_TITLES[key]}» недоступен: {unavailable[key]}.")

    multi, single = split_by_history(results)
    if multi and single:
        for title, subset in (("Многоходовые вопросы", multi), ("Самостоятельные вопросы", single)):
            lines.append(f"\n▶ {title} ({len(subset)})")
            lines.extend(_mode_blocks(aggregate_retrieval(subset, mode_keys), mode_keys))
    else:
        lines.extend(_mode_blocks(aggregates, mode_keys))

    rewrites = [r for r in results if r.rewrite_query or r.rewrite_failure]
    if rewrites:
        done = [r for r in rewrites if r.rewrite_query]
        average = sum(r.rewrite_seconds for r in rewrites) / len(rewrites)
        lines.append(
            f"\nПереписывание: удалось {len(done)} из {len(rewrites)}, среднее время "
            f"{average:.1f} с."
        )
    context_rewrites = [r for r in results if r.context_rewrite_query or r.context_rewrite_failure]
    if context_rewrites:
        done = [r for r in context_rewrites if r.context_rewrite_query]
        average = sum(r.context_rewrite_seconds for r in context_rewrites) / len(context_rewrites)
        lines.append(
            f"Переписывание с историей диалога: удалось {len(done)} из {len(context_rewrites)}, "
            f"среднее время {average:.1f} с."
        )
    if excluded:
        lines.append(
            "⚠️ Ожидаемого документа нет в индексе (исключены из метрик) на вопросах: "
            + ", ".join(str(n) for n in excluded)
            + "."
        )
    if report.get("index_missing"):
        lines.append("Не найдены в индексе: " + "; ".join(report["index_missing"]) + ".")
    lines.append(f"\n{ev.DISCLAIMER}\nДетали вопроса: /research_rag_compare_report <номер>.")
    return "\n".join(lines)


def _title(question: ev.Question) -> str:
    text = question.question
    return text if len(text) <= 38 else text[:37] + "…"


def _mode_cell(question: ev.Question, mode: RetrievalResult | None) -> str:
    if mode is None:
        return "—"
    if mode.error:
        return "сбой"
    position = first_hit_position(question.sources, mode.titles)
    if not mode.chunks:
        return "пусто"
    return f"#{position}" if position is not None else "мимо"


def format_retrieval_table(report: dict) -> str:
    results = [QuestionRetrieval.from_dict(q) for q in report.get("questions", [])]
    mode_keys = [m for m in report.get("modes", []) if m in RETRIEVAL_MODES]
    header = " · ".join(MODE_TITLES[m] for m in mode_keys)
    lines = [f"📋 По вопросам: позиция первого попадания ({header})\n"]
    for r in results:
        cells = " · ".join(_mode_cell(r.question, r.modes.get(m)) for m in mode_keys)
        note = " ⚠️ источника нет в индексе" if r.source_missing else ""
        lines.append(f"{r.question.number}. {_title(r.question)}\n   {cells}{note}")
    return "\n".join(lines)


def format_retrieval_detail(report: dict, number: int) -> str | None:
    """Детали вопроса режима «только поиск»; None — вопроса с таким номером нет."""
    for raw in report.get("questions", []):
        if raw.get("number") == number:
            result = QuestionRetrieval.from_dict(raw)
            break
    else:
        return None
    q = result.question
    lines = [
        f"❓ Вопрос {q.number} ({ev.KIND_LABELS.get(q.kind, q.kind)}): {q.question}",
        "\nОжидаемые источники: " + "; ".join(q.sources),
    ]
    if q.history:
        lines.append("История диалога (предыдущие вопросы): " + " → ".join(q.history))
    if result.missing_sources:
        lines.append("⚠️ Нет в индексе: " + "; ".join(result.missing_sources))
    if result.rewrite_query:
        lines.append(
            f"\nПереписанный запрос ({result.rewrite_seconds:.1f} с): {result.rewrite_query}"
        )
    elif result.rewrite_failure:
        lines.append(f"\nПереписывание не удалось: {result.rewrite_failure}")
    if result.context_rewrite_query:
        lines.append(
            f"Переписанный запрос с историей ({result.context_rewrite_seconds:.1f} с): "
            f"{result.context_rewrite_query}"
        )
    elif result.context_rewrite_failure:
        lines.append(f"Переписывание с историей не удалось: {result.context_rewrite_failure}")
    for key in [m for m in report.get("modes", []) if m in RETRIEVAL_MODES]:
        mode = result.modes.get(key)
        lines.append(f"\n— {MODE_TITLES[key].capitalize()} —")
        if mode is None:
            lines.append("Нет данных.")
        elif mode.error:
            lines.append(f"Сбой поиска: {mode.error}")
        elif not mode.chunks:
            lines.append(f"Материалов нет (кандидатов найдено: {mode.candidates}).")
        else:
            lines.append(f"Кандидатов найдено: {mode.candidates}, в выдаче: {len(mode.chunks)}.")
            for position, chunk in enumerate(mode.chunks, start=1):
                mark = "✅" if ev.source_hit(q.sources, [chunk.get("title", "")]) else "▫️"
                score = float(chunk.get("score", 0.0))
                lines.append(
                    f"{mark} {position}. {chunk.get('title', '?')} — фрагмент "
                    f"{chunk.get('chunk_index', '?')}, близость {score:.2f}"
                )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Режимы поиска в сравнении ответов (уровень answers)
# --------------------------------------------------------------------------- #


def aggregate_variants(results: list[ev.QuestionResult], mode_keys: list[str]) -> dict[str, dict]:
    """Покрытие ожидаемых фактов у дополнительных режимов поиска (QuestionResult.variants) в
    сравнении с режимом «без RAG». В расчёт идут вопросы, оценённые и в режиме «без RAG», и в
    этом режиме, без сбоя поиска. better/equal/worse — относительно «без RAG»."""
    aggregates: dict[str, dict] = {}
    for key in mode_keys:
        rows = []
        for r in results:
            mode = r.variants.get(key)
            if (
                mode is not None
                and not mode.error
                and not mode.search_failed
                and mode.coverage is not None
                and r.off.coverage is not None
                and not r.off.error
            ):
                rows.append((r, mode))
        counts = {"better": 0, "equal": 0, "worse": 0}
        for r, mode in rows:
            counts[ev.compare_coverage(r.off.coverage, mode.coverage)] += 1
        count = len(rows)
        aggregates[key] = {
            "scored": count,
            "avg": sum(m.coverage for _, m in rows) / count if count else None,
            "avg_off": sum(r.off.coverage for r, _ in rows) / count if count else None,
            "hit": sum(1 for r, m in rows if ev.hit(r.question, m)),
            "fired": sum(1 for _, m in rows if ev.fired(m)),
            "seconds": sum(m.elapsed for _, m in rows),
            **counts,
        }
    return aggregates


def format_variants_summary(report: dict) -> str | None:
    """Блок «режимы поиска» для сводки ответов; None — дополнительных режимов в прогоне не было."""
    mode_keys = [m for m in report.get("variant_modes", []) if m in RETRIEVAL_MODES]
    unavailable = report.get("unavailable_modes") or {}
    if not mode_keys and not unavailable:
        return None
    results = [ev.QuestionResult.from_dict(q) for q in report.get("questions", [])]
    aggregates = aggregate_variants(results, mode_keys)
    lines = ["🔀 Режимы поиска в ответах (покрытие ожидаемых фактов, относительно «без RAG»)"]
    for key in RETRIEVAL_MODES:
        if key in unavailable:
            lines.append(f"⚠️ «{MODE_TITLES[key]}» недоступен: {unavailable[key]}.")
    for key in mode_keys:
        agg = aggregates[key]
        if not agg["scored"]:
            lines.append(f"\n• {MODE_TITLES[key]}: нет оценённых ответов.")
            continue
        lines.append(
            f"\n• {MODE_TITLES[key]} (по {agg['scored']} вопросам)\n"
            f"  покрытие: {_pct(agg['avg'])} (без RAG: {_pct(agg['avg_off'])})\n"
            f"  лучше: {agg['better']}, равно: {agg['equal']}, хуже: {agg['worse']}\n"
            f"  поиск сработал в {agg['fired']}, попал в ожидаемый документ в {agg['hit']}; "
            f"время ответов {agg['seconds']:.0f} с"
        )
    return "\n".join(lines)
