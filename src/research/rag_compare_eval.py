"""Чистые правила и тексты сравнения ответов `/smart_agent` с выключенным и включённым
слоем rag (команда /research_rag_compare, design.md изменения add-rag-compare-command).

Отделён от research/rag_compare.py по принципу проекта (как agents/invariants.py,
agents/market_tools.py, agents/rag_context.py): здесь ПРАВИЛА и ТЕКСТЫ — загрузка и проверка
контрольного набора, запрос оценщику и разбор его ответа, покрытие ожидаемых фактов,
сопоставление источников, метрики поиска, агрегат прогона, тексты отчёта; а обращения к
SmartAgent и модели, фоновый прогон, Telegram и диск — там. Модуль не импортирует ни openai,
ни faiss, ни config, ни Telegram, поэтому все ветвления проверяются тестами
(tests/test_rag_compare_eval.py) без сети.

Отчёт прогона — обычный dict (его пишет и читает rag_compare.py как JSON): настройки прогона,
список заголовков документов индекса, которых не хватает набору, и результаты вопросов
(`QuestionResult.to_dict()`). Агрегат считается из результатов на лету, а не хранится, поэтому
старые отчёты читаются теми же функциями.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

KIND_OUT_OF_CORPUS = "out_of_corpus"  # только с expect_abstain (вопрос вне корпуса)
KINDS = ("corpus_specific", "concept", "post_cutoff")
KIND_LABELS = {
    KIND_OUT_OF_CORPUS: "вне корпуса",
    "corpus_specific": "по специфике корпуса",
    "concept": "по общему понятию",
    "post_cutoff": "после даты знаний модели",
}

MODE_OFF = "off"
MODE_ON = "on"
MODE_LABELS = {MODE_OFF: "без RAG", MODE_ON: "с RAG"}

_EPSILON = 1e-9
_NOTE_MAX_CHARS = 200
_TITLE_MAX_CHARS = 38
_REASON_MAX_CHARS = 120

SEARCH_FAILED_REASON = "поиск материалов не удался"
QUESTION_TIMEOUT_REASON = "превышен лимит времени вопроса"

STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"

# Этапы обработки одного вопроса — в том порядке, в котором они идут; по ним строится прогресс.
STAGE_OFF_ANSWER = "off_answer"
STAGE_OFF_JUDGE = "off_judge"
STAGE_ON_ANSWER = "on_answer"
STAGE_ON_JUDGE = "on_judge"
STAGE_ON_CITATIONS = "on_citations"
STAGES = (STAGE_OFF_ANSWER, STAGE_OFF_JUDGE, STAGE_ON_ANSWER, STAGE_ON_JUDGE, STAGE_ON_CITATIONS)
STAGE_LABELS = {
    STAGE_OFF_ANSWER: "ответ без RAG",
    STAGE_OFF_JUDGE: "оценка ответа без RAG",
    STAGE_ON_ANSWER: "ответ с RAG",
    STAGE_ON_JUDGE: "оценка ответа с RAG",
    STAGE_ON_CITATIONS: "оценка цитат ответа с RAG",
}
DISCLAIMER = (
    "Оценка ответов — вызов той же модели по чек-листу фактов, а один прогон недетерминирован: "
    "это грубый ориентир на 10 вопросах, а не статистика."
)


class QuestionSetError(ValueError):
    """Контрольный набор недопустим — команда не начинает прогон и сообщает об этом."""


@dataclass(frozen=True)
class Question:
    number: int  # с единицы, как в команде чтения деталей
    question: str
    kind: str
    facts: tuple[str, ...]
    sources: tuple[str, ...]  # префиксы заголовков документов индекса
    # Предыдущие вопросы диалога (от старых к новым), после которых задан этот вопрос; пусто —
    # самостоятельный вопрос (design.md изменения add-rag-rewrite-dialog-context, решение 6).
    history: tuple[str, ...] = ()
    # Вопрос вне корпуса: ожидание — режим «не знаю» (add-rag-citations-and-abstain, решение 6);
    # факты и источники не требуются, ответ по фактам не оценивается.
    expect_abstain: bool = False


def parse_questions(raw: Any) -> list[Question]:
    """Проверяет и разбирает набор: список непустой, у каждого вопроса непустые формулировка,
    факты и источники, тип из KINDS. Вопрос с `expect_abstain: true` (вне корпуса) требует
    только формулировку: тип по умолчанию KIND_OUT_OF_CORPUS, факты и источники не нужны.
    Любое нарушение — QuestionSetError с номером вопроса."""
    if not isinstance(raw, list) or not raw:
        raise QuestionSetError("набор пуст или не является списком вопросов")
    questions: list[Question] = []
    for number, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise QuestionSetError(f"вопрос {number}: ожидался объект")
        text = item.get("question")
        if not isinstance(text, str) or not text.strip():
            raise QuestionSetError(f"вопрос {number}: нет формулировки")
        expect_abstain = item.get("expect_abstain", False)
        if not isinstance(expect_abstain, bool):
            raise QuestionSetError(f"вопрос {number}: expect_abstain должен быть true или false")
        kind = item.get("kind")
        if expect_abstain and kind is None:
            kind = KIND_OUT_OF_CORPUS
        if kind not in KINDS and not (expect_abstain and kind == KIND_OUT_OF_CORPUS):
            raise QuestionSetError(f"вопрос {number}: недопустимый тип {kind!r}")
        facts = _non_empty_strings(item.get("facts"))
        sources = _non_empty_strings(item.get("sources"))
        if not expect_abstain:
            if not facts:
                raise QuestionSetError(f"вопрос {number}: нет ожидаемых фактов")
            if not sources:
                raise QuestionSetError(f"вопрос {number}: нет ожидаемых источников")
        history = _parse_history(item.get("history"), number)
        questions.append(
            Question(
                number=number,
                question=text.strip(),
                kind=kind,
                facts=facts,
                sources=sources,
                history=history,
                expect_abstain=expect_abstain,
            )
        )
    return questions


def _parse_history(value: Any, number: int) -> tuple[str, ...]:
    """Необязательное поле `history` вопроса: список непустых строк (предыдущие вопросы диалога
    от старых к новым). Нет поля или null — самостоятельный вопрос; не список или пустая строка
    в списке — QuestionSetError с номером вопроса (молча выбросить реплику значило бы изменить
    контекст без ведома автора набора)."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise QuestionSetError(f"вопрос {number}: history должен быть списком строк")
    items: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise QuestionSetError(
                f"вопрос {number}: history содержит пустую или не строковую реплику"
            )
        items.append(entry.strip())
    return tuple(items)


def _non_empty_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items = tuple(v.strip() for v in value if isinstance(v, str) and v.strip())
    # Частично пустой список — тоже ошибка набора: молча выбросить факт значило бы занизить
    # или завысить покрытие без ведома автора набора.
    return items if len(items) == len(value) else ()


def split_multi_turn(questions: list[Question]) -> tuple[list[Question], list[Question]]:
    """(самостоятельные, многоходовые) вопросы набора. На уровне сравнения ответов многоходовые
    (с `history`) не выполняются: ответ на продолжение без настоящей предыстории с ответами
    модели несравним, такие вопросы проверяются уровнем search (design.md изменения
    add-rag-rewrite-dialog-context, решение 8). Номера вопросов не меняются."""
    standalone = [q for q in questions if not q.history]
    multi_turn = [q for q in questions if q.history]
    return standalone, multi_turn


def load_questions(path: str) -> list[Question]:
    """Читает набор из JSON-файла; нечитаемый файл или не-JSON — тоже QuestionSetError."""
    try:
        with open(path, encoding="utf-8") as file:
            raw = json.load(file)
    except FileNotFoundError as exc:
        raise QuestionSetError(f"файл набора не найден: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise QuestionSetError(f"не удалось прочитать набор: {exc}") from exc
    return parse_questions(raw)


# --------------------------------------------------------------------------- #
# Оценка ответа по чек-листу фактов (вслепую по режиму)
# --------------------------------------------------------------------------- #


def build_judge_user_content(question: str, facts: tuple[str, ...] | list[str], answer: str) -> str:
    """Запрос оценщику: вопрос, пронумерованные факты, ответ. О режиме (с материалами или
    без) здесь нет ничего — оценка вслепую по построению."""
    numbered = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, start=1))
    return (
        f"Вопрос:\n{question}\n\n"
        f"Ожидаемые факты:\n{numbered}\n\n"
        f"Ответ ассистента:\n<<<\n{answer}\n>>>"
    )


@dataclass(frozen=True)
class Verdict:
    """present[i] — присутствует ли факт i+1 из списка вопроса."""

    present: tuple[bool, ...]
    contradicts: bool = False
    note: str = ""

    @property
    def coverage(self) -> float:
        return sum(self.present) / len(self.present) if self.present else 0.0

    def to_dict(self) -> dict:
        return {"present": list(self.present), "contradicts": self.contradicts, "note": self.note}

    @staticmethod
    def from_dict(data: dict) -> Verdict:
        return Verdict(
            present=tuple(bool(v) for v in data.get("present", [])),
            contradicts=bool(data.get("contradicts", False)),
            note=str(data.get("note", "")),
        )


def parse_judge_response(data: Any, fact_count: int) -> Verdict | None:
    """Разбирает JSON оценщика. None — ответ непригоден (не объект, нет массива `facts`):
    такой ответ не оценён. Пропущенный факт считается отсутствующим, лишние и повторные
    номера игнорируются (первая запись по номеру побеждает), `contradicts` истинно только при
    явном true."""
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        return None
    flags: dict[int, bool] = {}
    for entry in data["facts"]:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if 1 <= index <= fact_count and index not in flags:
            flags[index] = entry.get("present") is True
    present = tuple(flags.get(i, False) for i in range(1, fact_count + 1))
    note = data.get("note")
    return Verdict(
        present=present,
        contradicts=data.get("contradicts") is True,
        note=note.strip()[:_NOTE_MAX_CHARS] if isinstance(note, str) else "",
    )


# --------------------------------------------------------------------------- #
# Оценка совпадения смысла ответа и проверенных цитат (вслепую по режиму)
# --------------------------------------------------------------------------- #

CITATION_SUPPORTED = "supported"
CITATION_PARTIAL = "partial"
CITATION_UNSUPPORTED = "unsupported"
CITATION_VERDICTS = (CITATION_SUPPORTED, CITATION_PARTIAL, CITATION_UNSUPPORTED)
# Значение settings["citation_judge"] у отчётов, где оценщик видел полный текст фрагментов; в
# отчётах без ключа оценка шла только по цитатам (вердикты несопоставимы).
CITATION_JUDGE_FRAGMENTS = "fragments"
CITATION_VERDICT_LABELS = {
    CITATION_SUPPORTED: "подтверждается",
    CITATION_PARTIAL: "подтверждается частично",
    CITATION_UNSUPPORTED: "не подтверждается",
}
_MAX_UNSUPPORTED_CLAIMS = 5


# Предел длины одного фрагмента во входе оценщика цитат (размер чанка зависит от стратегии).
CITATION_JUDGE_FRAGMENT_MAX_CHARS = 3000
_TRUNCATED_MARK = " […фрагмент усечён]"


def unique_fragments(citations: list) -> list[str]:
    """Тексты фрагментов проверенных цитат (`VerifiedCitation.fragment`) без повторов, в порядке
    первого упоминания; фрагмент определяется по chunk_id, а без него — по заголовку и номеру."""
    seen: set = set()
    result: list[str] = []
    for cite in citations:
        key = cite.source.chunk_id or (cite.source.title, cite.source.chunk_index)
        if key in seen or not cite.fragment:
            continue
        seen.add(key)
        result.append(cite.fragment)
    return result


def _judge_fragment(text: str) -> str:
    text = text.strip()
    if len(text) <= CITATION_JUDGE_FRAGMENT_MAX_CHARS:
        return text
    return text[:CITATION_JUDGE_FRAGMENT_MAX_CHARS].rstrip() + _TRUNCATED_MARK


def build_citation_judge_user_content(
    question: str, answer: str, quotes: list[str], fragments: list[str]
) -> str:
    """Запрос оценщику: вопрос, пронумерованные фрагменты (полный текст, источник истины),
    пронумерованные проверенные цитаты (указатели на места фрагментов) и ответ. Указания на режим
    (с материалами или без) нет по построению."""
    numbered_fragments = "\n\n".join(
        f"[{i}]\n{_judge_fragment(fragment)}" for i, fragment in enumerate(fragments, start=1)
    )
    numbered_quotes = "\n".join(f"{i}. {quote}" for i, quote in enumerate(quotes, start=1))
    return (
        f"Вопрос:\n{question}\n\n"
        f"Фрагменты:\n{numbered_fragments}\n\n"
        f"Цитаты:\n{numbered_quotes}\n\n"
        f"Ответ ассистента:\n<<<\n{answer}\n>>>"
    )


@dataclass(frozen=True)
class CitationVerdict:
    """verdict — один из CITATION_VERDICTS; unsupported — утверждения ответа без опоры в
    цитатах (по словам оценщика)."""

    verdict: str
    unsupported: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "unsupported": list(self.unsupported)}

    @staticmethod
    def from_dict(data: dict) -> CitationVerdict:
        return CitationVerdict(
            verdict=str(data.get("verdict", "")),
            unsupported=tuple(str(c) for c in data.get("unsupported", [])),
        )


def parse_citation_judge_response(data: Any) -> CitationVerdict | None:
    """Разбирает JSON оценщика цитат. None — непригоден (не объект или вердикт вне
    CITATION_VERDICTS). Утверждения без опоры — только непустые строки, не более
    _MAX_UNSUPPORTED_CLAIMS, усечённые."""
    if not isinstance(data, dict) or data.get("verdict") not in CITATION_VERDICTS:
        return None
    claims = data.get("unsupported_claims")
    unsupported = tuple(
        _short(c)
        for c in (claims if isinstance(claims, list) else [])
        if isinstance(c, str) and c.strip()
    )[:_MAX_UNSUPPORTED_CLAIMS]
    return CitationVerdict(verdict=data["verdict"], unsupported=unsupported)


# --------------------------------------------------------------------------- #
# Результаты вопросов
# --------------------------------------------------------------------------- #


@dataclass
class ModeResult:
    """Результат одного режима по одному вопросу. error — причина сбоя получения ответа;
    judge_error — причина, по которой ответ не оценён; search_failed — в режиме с RAG поиск
    не удался (ответ получен без материалов, сравнение по такому вопросу бессмысленно)."""

    answer: str | None = None
    error: str | None = None
    elapsed: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    llm_calls: int = 0
    warnings: list[str] = field(default_factory=list)
    rag_sources: list[dict] = field(default_factory=list)  # {"title","chunk_index","score"}
    search_failed: bool = False
    verdict: Verdict | None = None
    judge_error: str | None = None
    # Цитаты и «не знаю» (add-rag-citations-and-abstain): проверенные цитаты ответа
    # ({"quote","title","chunk_index","chunk_id"}), сколько цитат написала модель, признаки
    # «цитаты не подтверждены» и отказа, оценка совпадения смысла ответа и цитат.
    citations: list[dict] = field(default_factory=list)
    citations_written: int = 0
    citations_unverified: bool = False
    abstained: bool = False
    citation_verdict: CitationVerdict | None = None
    citation_judge_error: str | None = None
    # Тексты фрагментов проверенных цитат (без повторов) — только на время прогона, для оценщика
    # цитат. В отчёт НЕ пишутся (to_dict их убирает) и из отчёта не читаются.
    fragments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("fragments", None)
        data["verdict"] = self.verdict.to_dict() if self.verdict else None
        data["citation_verdict"] = (
            self.citation_verdict.to_dict() if self.citation_verdict else None
        )
        return data

    @staticmethod
    def from_dict(data: dict) -> ModeResult:
        verdict = data.get("verdict")
        citation_verdict = data.get("citation_verdict")
        return ModeResult(
            answer=data.get("answer"),
            error=data.get("error"),
            elapsed=float(data.get("elapsed", 0.0)),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            llm_calls=int(data.get("llm_calls", 0)),
            warnings=list(data.get("warnings", [])),
            rag_sources=list(data.get("rag_sources", [])),
            search_failed=bool(data.get("search_failed", False)),
            verdict=Verdict.from_dict(verdict) if isinstance(verdict, dict) else None,
            judge_error=data.get("judge_error"),
            citations=list(data.get("citations", [])),
            citations_written=int(data.get("citations_written", 0)),
            citations_unverified=bool(data.get("citations_unverified", False)),
            abstained=bool(data.get("abstained", False)),
            citation_verdict=(
                CitationVerdict.from_dict(citation_verdict)
                if isinstance(citation_verdict, dict)
                else None
            ),
            citation_judge_error=data.get("citation_judge_error"),
        )

    @property
    def coverage(self) -> float | None:
        return self.verdict.coverage if self.verdict else None


@dataclass
class QuestionResult:
    question: Question
    off: ModeResult
    on: ModeResult
    missing_sources: list[str] = field(default_factory=list)  # ожидаемые, которых нет в индексе
    # Дополнительные режимы поиска (research/rag_modes_eval.py, уровень answers): ключ режима →
    # ответ и оценка в этом режиме. Пусто — прогон «без RAG / с RAG», как до этого поля.
    variants: dict[str, ModeResult] = field(default_factory=dict)

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
            "expect_abstain": q.expect_abstain,
            "missing_sources": list(self.missing_sources),
            "off": self.off.to_dict(),
            "on": self.on.to_dict(),
            "variants": {key: mode.to_dict() for key, mode in self.variants.items()},
        }

    @staticmethod
    def from_dict(data: dict) -> QuestionResult:
        question = Question(
            number=int(data["number"]),
            question=data["question"],
            kind=data["kind"],
            facts=tuple(data["facts"]),
            sources=tuple(data["sources"]),
            history=tuple(data.get("history") or ()),
            expect_abstain=bool(data.get("expect_abstain", False)),
        )
        return QuestionResult(
            question=question,
            off=ModeResult.from_dict(data.get("off", {})),
            on=ModeResult.from_dict(data.get("on", {})),
            missing_sources=list(data.get("missing_sources", [])),
            variants={
                key: ModeResult.from_dict(value)
                for key, value in (data.get("variants") or {}).items()
            },
        )


# --------------------------------------------------------------------------- #
# Источники, метрики поиска, сравнение
# --------------------------------------------------------------------------- #


def sources_not_indexed(
    expected: tuple[str, ...] | list[str], index_titles: list[str]
) -> list[str]:
    """Ожидаемые префиксы, ни с одним из которых не начинается заголовок документа индекса
    (без учёта регистра) — то есть документ пропущен при индексации."""
    titles = [t.casefold() for t in index_titles]
    return [p for p in expected if not any(t.startswith(p.casefold()) for t in titles)]


def source_hit(expected: tuple[str, ...] | list[str], used_titles: list[str]) -> bool:
    """Попал ли хотя бы один использованный фрагмент в ожидаемый документ."""
    prefixes = [p.casefold() for p in expected]
    return any(title.casefold().startswith(p) for title in used_titles for p in prefixes)


def fired(mode: ModeResult) -> bool:
    """Сработал ли поиск: хотя бы один фрагмент прошёл порог и попал в запрос."""
    return bool(mode.rag_sources)


def hit(question: Question, mode: ModeResult) -> bool:
    return source_hit(question.sources, [s.get("title", "") for s in mode.rag_sources])


def compare_coverage(off_coverage: float, on_coverage: float) -> str:
    """'better' — с RAG покрытие выше, 'worse' — ниже, иначе 'equal'."""
    if on_coverage > off_coverage + _EPSILON:
        return "better"
    if on_coverage < off_coverage - _EPSILON:
        return "worse"
    return "equal"


def citation_eligible(result: QuestionResult) -> bool:
    """Участвует ли вопрос в метриках источников и цитат: вопрос по корпусу, ответ с RAG получен,
    поиск не упал и бот не ответил режимом «не знаю» (ложный отказ считается отдельно)."""
    on = result.on
    return (
        not result.question.expect_abstain
        and on.answer is not None
        and not on.error
        and not on.search_failed
        and not on.abstained
    )


def exclusion_reason(result: QuestionResult) -> str | None:
    """Почему вопрос не участвует в средних и счётчиках сравнения; None — участвует. Нужны
    оценённые ответы ОБОИХ режимов, а сбой поиска в режиме с RAG делает сравнение пустым:
    ответ получен фактически без материалов."""
    for mode_key, mode in ((MODE_OFF, result.off), (MODE_ON, result.on)):
        label = MODE_LABELS[mode_key]
        if mode.error:
            return f"ответ {label}: {_short(mode.error)}"
    if result.on.search_failed:
        return SEARCH_FAILED_REASON
    for mode_key, mode in ((MODE_OFF, result.off), (MODE_ON, result.on)):
        if mode.verdict is None:
            label = MODE_LABELS[mode_key]
            return f"оценка {label}: {_short(mode.judge_error or 'нет оценки')}"
    return None


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _REASON_MAX_CHARS else text[: _REASON_MAX_CHARS - 1] + "…"


# --------------------------------------------------------------------------- #
# Жёсткий предел времени обращения (по часам)
# --------------------------------------------------------------------------- #

T = TypeVar("T")


class DeadlineExceeded(Exception):
    """Обращение не завершилось за отведённое время."""


def call_with_deadline(fn: Callable[[], T], seconds: float) -> T:
    """Выполняет `fn()` в отдельном потоке и ждёт результата не дольше `seconds` ПО ЧАСАМ.

    Таймауты HTTP-клиента этого не гарантируют: это паузы между чтениями, а не предел общего
    времени, и провайдер, держащий соединение (DeepSeek под нагрузкой шлёт пустые строки
    keep-alive, пока запрос стоит в очереди), может «ответить» через ~10 минут при таймауте
    в 60 секунд — так и случилось в живом прогоне. По истечении времени бросает
    DeadlineExceeded; поток-сирота (daemon) доживает сам, когда сервер закроет соединение, а
    его результат или ошибка отбрасываются. Исключение, брошенное `fn`, пробрасывается."""
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — передаётся вызывающему
            box["error"] = exc

    thread = threading.Thread(target=runner, name="rag-compare-call", daemon=True)
    thread.start()
    thread.join(max(seconds, 0.001))
    if thread.is_alive():
        raise DeadlineExceeded(f"нет ответа за {seconds:.0f} с")
    if "error" in box:
        raise box["error"]
    return box["value"]


# --------------------------------------------------------------------------- #
# Проваленные вопросы и остановка прогона
# --------------------------------------------------------------------------- #


def question_failed(result: QuestionResult) -> bool:
    """Провалился ли вопрос: из-за сбоя (провайдера, оценщика) или превышения времени не получен
    ответ или оценка хотя бы одного этапа. Сбой ПОИСКА материалов проваленным вопросом не
    считается: ответы получены, а такой вопрос и так исключается из сравнения (см.
    exclusion_reason). У вопроса вне корпуса оценки по фактам нет — проваливается только сбой
    ответа; сбой оценщика цитат проваливает вопрос, как и сбой оценщика фактов."""
    if result.question.expect_abstain:
        return bool(result.on.error)
    return bool(result.on.citation_judge_error) or any(
        mode.error or mode.judge_error or (mode.answer is not None and mode.verdict is None)
        for mode in (result.off, result.on)
    )


def consecutive_failures(results: list[QuestionResult]) -> int:
    """Сколько проваленных вопросов подряд в конце списка; успешный вопрос сбрасывает счёт."""
    count = 0
    for result in reversed(results):
        if not question_failed(result):
            break
        count += 1
    return count


def should_stop(results: list[QuestionResult], limit: int) -> bool:
    """Пора ли остановить прогон: подряд провалилось не меньше `limit` вопросов."""
    return limit > 0 and consecutive_failures(results) >= limit


# Имя файла отчёта: UTC-время прогона (его пишет research/rag_compare.py). Файлы набора вопросов
# и калибровки лежат в том же каталоге data/rag_eval/ и тоже заканчиваются на .json, поэтому
# «последний отчёт» выбирается только среди файлов с таким именем.
_REPORT_NAME = re.compile(r"^\d{8}T\d{6}Z\.json$")


def latest_report_name(names: list[str]) -> str | None:
    """Имя последнего отчёта среди имён файлов каталога (по времени в имени); None — отчётов нет.
    Файлы другого вида (наборы вопросов, копии, временные файлы) игнорируются."""
    reports = sorted(name for name in names if _REPORT_NAME.match(name))
    return reports[-1] if reports else None


def build_report(
    *,
    started_at: str,
    finished_at: str,
    settings: dict,
    results: list[QuestionResult],
    planned: int,
    index_missing: list[str] | None = None,
    status: str = STATUS_COMPLETED,
    stop_reason: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Отчёт прогона (его пишет на диск rag_compare.py). `planned` — число вопросов набора;
    у остановленного прогона обработанных меньше. Отчёты без `status`/`planned` (написанные до
    появления остановки) читаются как завершённые. `extra` — дополнительные поля верхнего
    уровня (режимы поиска прогона: `variant_modes`, `unavailable_modes`); их читают функции
    research/rag_modes_eval.py, старые отчёты без них остаются корректными."""
    return {
        **(extra or {}),
        "status": status,
        "stop_reason": stop_reason,
        "planned": planned,
        "started_at": started_at,
        "finished_at": finished_at,
        "settings": settings,
        "index_missing": sorted(index_missing or []),
        "questions": [r.to_dict() for r in results],
    }


def progress_text(
    done: int,
    total: int,
    stage: str | None = None,
    question: str | None = None,
    stage_label: str | None = None,
) -> str:
    """Текст сообщения с прогрессом. Без этапа — общее число обработанных вопросов; с этапом —
    номер текущего вопроса и этап (design.md, решение 5). `stage_label` — подпись этапа, которого
    нет в STAGE_LABELS (режимы поиска, research/rag_modes_eval.py)."""
    if stage is None:
        return f"⏳ Сравнение без RAG и с RAG: обработано {done} из {total} вопросов."
    text = (
        f"⏳ Вопрос {done + 1} из {total} · {stage_label or STAGE_LABELS.get(stage, stage)}\n"
        f"Обработано вопросов: {done}."
    )
    return f"{text}\n{question}" if question else text


# --------------------------------------------------------------------------- #
# Агрегат прогона
# --------------------------------------------------------------------------- #


def aggregate(results: list[QuestionResult]) -> dict:
    """Сводные числа прогона. В средних и счётчиках better/equal/worse участвуют только
    вопросы без причины исключения (exclusion_reason); `unavailable_modes` — режимы, по
    которым не получено ни одного оценённого ответа (сравнение невозможно)."""
    scored: list[QuestionResult] = []
    excluded: list[dict] = []
    # Вопросы вне корпуса (expect_abstain) в сравнении по фактам не участвуют: у них нет ни
    # фактов, ни ответа без RAG; их считают метрики режима «не знаю» ниже.
    fact_results = [r for r in results if not r.question.expect_abstain]
    for result in fact_results:
        reason = exclusion_reason(result)
        if reason is None:
            scored.append(result)
        else:
            excluded.append({"number": result.question.number, "reason": reason})

    off_cov = [r.off.coverage for r in scored]
    on_cov = [r.on.coverage for r in scored]
    verdicts = {"better": 0, "equal": 0, "worse": 0}
    for r in scored:
        verdicts[compare_coverage(r.off.coverage, r.on.coverage)] += 1

    on_answered = [r for r in fact_results if r.on.answer is not None and not r.on.error]
    hit_candidates = [r for r in on_answered if not r.source_missing and not r.on.search_failed]

    def total(values) -> float | int:
        return sum(v for v in values if v is not None)

    def count_judged(mode_key: str) -> int:
        return sum(1 for r in fact_results if getattr(r, mode_key).verdict is not None)

    eligible = [r for r in fact_results if citation_eligible(r)]
    ratios = [
        len(r.on.citations) / r.on.citations_written for r in eligible if r.on.citations_written > 0
    ]
    cit_verdicts = {key: 0 for key in CITATION_VERDICTS}
    for r in eligible:
        if r.on.citation_verdict and r.on.citation_verdict.verdict in cit_verdicts:
            cit_verdicts[r.on.citation_verdict.verdict] += 1
    abstain_questions = [r for r in results if r.question.expect_abstain]

    return {
        "total": len(results),
        "facts_total": len(fact_results),
        "cit_eligible": len(eligible),
        "cit_sources": sum(1 for r in eligible if r.on.rag_sources),
        "cit_quotes": sum(1 for r in eligible if r.on.citations),
        "cit_no_quotes": [r.question.number for r in eligible if not r.on.citations],
        "cit_avg_ratio": sum(ratios) / len(ratios) if ratios else None,
        "cit_ratio_answers": len(ratios),
        "cit_verdicts": cit_verdicts,
        "cit_judge_missing": sum(
            1 for r in eligible if r.on.citations and r.on.citation_verdict is None
        ),
        "abstain_expected": len(abstain_questions),
        "abstain_correct": sum(1 for r in abstain_questions if r.on.abstained),
        "abstain_wrong": [
            r.question.number
            for r in abstain_questions
            if not r.on.abstained and r.on.answer is not None and not r.on.error
        ],
        "abstain_errors": [r.question.number for r in abstain_questions if r.on.error],
        "false_abstain": [r.question.number for r in fact_results if r.on.abstained],
        "scored": len(scored),
        "excluded": excluded,
        "avg_off": sum(off_cov) / len(off_cov) if off_cov else None,
        "avg_on": sum(on_cov) / len(on_cov) if on_cov else None,
        "delta": (sum(on_cov) - sum(off_cov)) / len(scored) if scored else None,
        "better": verdicts["better"],
        "equal": verdicts["equal"],
        "worse": verdicts["worse"],
        "contradicts_off": sum(1 for r in scored if r.off.verdict.contradicts),
        "contradicts_on": sum(1 for r in scored if r.on.verdict.contradicts),
        "on_answered": len(on_answered),
        "fired": sum(1 for r in on_answered if fired(r.on)),
        "hit": sum(1 for r in hit_candidates if hit(r.question, r.on)),
        "hit_candidates": len(hit_candidates),
        "not_fired": [
            r.question.number
            for r in on_answered
            if not fired(r.on) and not r.on.search_failed and not r.on.abstained
        ],
        "search_failed": [r.question.number for r in fact_results if r.on.search_failed],
        "source_missing": [r.question.number for r in fact_results if r.source_missing],
        "tokens_off": total(
            (r.off.prompt_tokens or 0) + (r.off.completion_tokens or 0) for r in results
        ),
        "tokens_on": total(
            (r.on.prompt_tokens or 0) + (r.on.completion_tokens or 0) for r in results
        ),
        "time_off": total(r.off.elapsed for r in results),
        "time_on": total(r.on.elapsed for r in results),
        "unavailable_modes": [
            key for key in (MODE_OFF, MODE_ON) if fact_results and count_judged(key) == 0
        ],
    }


# --------------------------------------------------------------------------- #
# Тексты отчёта (обычный текст без разметки Telegram)
# --------------------------------------------------------------------------- #


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{round(value * 100)}%"


def _signed_pct(value: float | None) -> str:
    if value is None:
        return "—"
    points = round(value * 100)
    return f"{points:+d} п.п."


def _numbers(values: list[int]) -> str:
    return ", ".join(str(v) for v in values) if values else "нет"


def _settings_line(settings: dict) -> str:
    if not settings:
        return ""
    parts = [
        f"стратегия {settings.get('strategy', '?')}",
        f"top-k {settings.get('top_k', '?')}",
        f"порог {settings.get('min_score', '?')}",
    ]
    if settings.get("candidates"):
        parts.append(f"кандидатов {settings['candidates']}")
    if settings.get("model"):
        parts.append(f"модель {settings['model']}")
    if settings.get("rewrite_provider"):
        rewrite = settings["rewrite_provider"]
        if settings.get("rewrite_model"):
            rewrite += f"/{settings['rewrite_model']}"
        parts.append(f"переписывание {rewrite}")
    return "Настройки прогона: " + ", ".join(parts) + "."


def format_summary(report: dict) -> str:
    results = [QuestionResult.from_dict(q) for q in report.get("questions", [])]
    agg = aggregate(results)
    lines = ["📊 Сравнение ответов /smart_agent: без RAG и с RAG"]
    if report.get("status") == STATUS_STOPPED:
        planned = report.get("planned", len(results))
        reason = report.get("stop_reason") or "причина не указана"
        lines.append(
            f"⛔ Прогон остановлен: {reason}. Обработано {len(results)} из {planned} вопросов; "
            "метрики ниже — только по обработанным."
        )
    if report.get("finished_at"):
        verb = "остановлен" if report.get("status") == STATUS_STOPPED else "завершён"
        lines.append(f"Прогон {verb}: {report['finished_at']} (UTC).")
    settings = _settings_line(report.get("settings", {}))
    if settings:
        lines.append(settings)

    if agg["unavailable_modes"]:
        names = ", ".join(MODE_LABELS[m] for m in agg["unavailable_modes"])
        lines.append(f"\n⚠️ Сравнение невозможно: нет оценённых ответов в режиме «{names}».")
    elif agg["facts_total"] == 0 and agg["abstain_expected"]:
        pass  # в наборе только вопросы вне корпуса — сравнения по фактам нет
    elif agg["scored"] == 0:
        lines.append(
            "\n⚠️ Сравнение невозможно: ни один вопрос не набрал оценённых ответов в обоих режимах."
        )
    else:
        lines.append(
            f"\nПокрытие ожидаемых фактов (по {agg['scored']} из {agg['facts_total']} вопросов):\n"
            f"• без RAG: {_pct(agg['avg_off'])}\n"
            f"• с RAG: {_pct(agg['avg_on'])}\n"
            f"• разница: {_signed_pct(agg['delta'])}\n"
            f"С RAG лучше: {agg['better']}, равно: {agg['equal']}, хуже: {agg['worse']}.\n"
            f"Противоречия ожиданию: без RAG — {agg['contradicts_off']}, "
            f"с RAG — {agg['contradicts_on']}."
        )

    lines.append(
        f"\nПоиск: сработал в {agg['fired']} из {agg['on_answered']} ответов с RAG; "
        f"попал в ожидаемый документ в {agg['hit']} из {agg['hit_candidates']}."
    )
    by_fragments = report.get("settings", {}).get("citation_judge") == CITATION_JUDGE_FRAGMENTS
    lines += _citation_summary_lines(agg, by_fragments=by_fragments)
    if agg["not_fired"]:
        lines.append(
            "Поиск не сработал (материалы не использованы) на вопросах: "
            f"{_numbers(agg['not_fired'])}."
        )
    if agg["source_missing"]:
        lines.append(
            "⚠️ Ожидаемого документа нет в индексе (пропущен при индексации) на вопросах: "
            f"{_numbers(agg['source_missing'])}."
        )
    if agg["search_failed"]:
        lines.append(f"⚠️ Поиск не удался на вопросах: {_numbers(agg['search_failed'])}.")
    if report.get("index_missing"):
        lines.append("Не найдены в индексе: " + "; ".join(report["index_missing"]) + ".")

    lines.append(
        f"\nТокены (контекст+ответ): без RAG — {agg['tokens_off']}, с RAG — {agg['tokens_on']}. "
        f"Время ответов: без RAG — {agg['time_off']:.0f} с, с RAG — {agg['time_on']:.0f} с."
    )
    if agg["excluded"]:
        details = "; ".join(f"{e['number']} ({e['reason']})" for e in agg["excluded"])
        lines.append(f"Исключено из расчёта: {len(agg['excluded'])} — {details}.")
    lines.append(f"\n{DISCLAIMER}\nДетали вопроса: /research_rag_compare_report <номер>.")
    return "\n".join(lines)


def _citation_summary_lines(agg: dict, by_fragments: bool = True) -> list[str]:
    """Блок сводки про источники, цитаты и режим «не знаю»; пусто, если оценивать нечего."""
    lines: list[str] = []
    eligible = agg["cit_eligible"]
    if eligible:
        verdicts = agg["cit_verdicts"]
        lines.append(
            f"\nИсточники и цитаты (ответы с RAG, по {eligible} вопросам по корпусу):\n"
            f"• с источниками: {agg['cit_sources']} из {eligible}\n"
            f"• с проверенными цитатами: {agg['cit_quotes']} из {eligible}\n"
            f"• средняя доля проверенных цитат среди написанных: {_pct(agg['cit_avg_ratio'])} "
            f"(по {agg['cit_ratio_answers']} ответам)\n"
            + (
                "• опора ответа на материалы (оценка по тексту фрагментов): "
                if by_fragments
                else "• смысл ответа и цитат (оценка только по цитатам, прежняя версия): "
            )
            + ", ".join(
                f"{CITATION_VERDICT_LABELS[key]} — {verdicts[key]}" for key in CITATION_VERDICTS
            )
            + (f"; без оценки — {agg['cit_judge_missing']}" if agg["cit_judge_missing"] else "")
        )
        if agg["cit_no_quotes"]:
            lines.append(
                f"Без проверенных цитат ответы на вопросы: {_numbers(agg['cit_no_quotes'])}."
            )
    if agg["abstain_expected"]:
        lines.append(
            f"\nРежим «не знаю»: верных отказов {agg['abstain_correct']} из "
            f"{agg['abstain_expected']} вопросов вне корпуса."
        )
        if agg["abstain_wrong"]:
            lines.append(f"Вместо отказа дан ответ на вопросах: {_numbers(agg['abstain_wrong'])}.")
        if agg["abstain_errors"]:
            lines.append(f"Сбой ответа на вопросах вне корпуса: {_numbers(agg['abstain_errors'])}.")
    if agg["false_abstain"]:
        lines.append(
            "⚠️ Ложный отказ («не знаю») на вопросах по корпусу: "
            f"{_numbers(agg['false_abstain'])}."
        )
    return lines


def _question_title(question: Question) -> str:
    text = question.question
    return text if len(text) <= _TITLE_MAX_CHARS else text[: _TITLE_MAX_CHARS - 1] + "…"


def _mode_cell(mode: ModeResult) -> str:
    if mode.error:
        return "сбой"
    if mode.coverage is None:
        return "н/о"
    return _pct(mode.coverage)


def _citation_cell(result: QuestionResult) -> str:
    """Хвост строки таблицы: источники, цитаты, смысл; для ложного отказа — пометка."""
    if result.on.abstained:
        return " · ⚠️ ложный отказ"
    if not citation_eligible(result):
        return ""
    on = result.on
    cell = f" · ист: {'✓' if on.rag_sources else '✗'} цит: {'✓' if on.citations else '✗'}"
    if on.citations_written:
        cell += f" ({len(on.citations)}/{on.citations_written})"
    if on.citation_verdict:
        cell += f" · смысл: {CITATION_VERDICT_LABELS.get(on.citation_verdict.verdict, '?')}"
    return cell


def format_table(report: dict) -> str:
    results = [QuestionResult.from_dict(q) for q in report.get("questions", [])]
    lines = ["📋 По вопросам: покрытие без RAG → с RAG, поиск\n"]
    for r in results:
        if r.question.expect_abstain:
            if r.on.error:
                outcome = "сбой ответа"
            elif r.on.abstained:
                outcome = "отказ ✓"
            else:
                outcome = "ответ вместо отказа ✗"
            lines.append(
                f"{r.question.number}. {_question_title(r.question)}\n   вне корпуса: {outcome}"
            )
            continue
        reason = exclusion_reason(r)
        if reason is None:
            trend = {"better": "▲", "equal": "=", "worse": "▼"}[
                compare_coverage(r.off.coverage, r.on.coverage)
            ]
        else:
            trend = "—"
        if r.on.search_failed:
            search = "поиск: сбой"
        elif r.on.error:
            search = "поиск: —"
        elif not fired(r.on):
            search = "поиск: не сработал"
        else:
            search = "поиск: попал" if hit(r.question, r.on) else "поиск: мимо"
        note = " ⚠️ источника нет в индексе" if r.source_missing else ""
        lines.append(
            f"{r.question.number}. {_question_title(r.question)}\n"
            f"   {_mode_cell(r.off)} → {_mode_cell(r.on)} {trend} · {search}{note}"
            f"{_citation_cell(r)}"
        )
    return "\n".join(lines)


def format_question_detail(
    report: dict, number: int, variant_titles: dict[str, str] | None = None
) -> str | None:
    """Детали вопроса; None — вопроса с таким номером нет. `variant_titles` — подписи
    дополнительных режимов поиска (ключ режима → название) для ответов в этих режимах."""
    for raw in report.get("questions", []):
        if raw.get("number") == number:
            result = QuestionResult.from_dict(raw)
            break
    else:
        return None

    q = result.question
    lines = [f"❓ Вопрос {q.number} ({KIND_LABELS[q.kind]}): {q.question}"]
    if q.expect_abstain:
        lines.append("\nОжидание: ответ «не знаю» (вопрос вне корпуса).")
    else:
        lines += [
            "\nОжидаемые факты:",
            *[f"{i}. {fact}" for i, fact in enumerate(q.facts, start=1)],
            "\nОжидаемые источники: " + "; ".join(q.sources),
        ]
    if q.history:
        lines.append(
            "История диалога (предыдущие вопросы): " + " → ".join(q.history)
        )
    if result.missing_sources:
        lines.append("⚠️ Нет в индексе: " + "; ".join(result.missing_sources))
    shown = [
        (MODE_OFF, MODE_LABELS[MODE_OFF].capitalize(), result.off),
        (MODE_ON, MODE_LABELS[MODE_ON].capitalize(), result.on),
    ]
    if q.expect_abstain:
        shown = shown[1:]  # ответ без RAG у вопроса вне корпуса не собирается
    for key, variant in result.variants.items():
        title = (variant_titles or {}).get(key, key)
        shown.append((f"variant:{key}", f"С RAG, режим поиска «{title}»", variant))
    for mode_key, label, mode in shown:
        lines.append(f"\n— {label} —")
        if mode.error:
            lines.append(f"Сбой ответа: {mode.error}")
            continue
        lines.append(mode.answer or "(пустой ответ)")
        if mode_key != MODE_OFF:
            if mode.search_failed:
                lines.append("⚠️ Поиск материалов не удался — ответ получен без них.")
            elif mode.rag_sources:
                lines.append("Использованные фрагменты:")
                lines.extend(
                    f"• {s.get('title', '?')} — фрагмент {s.get('chunk_index', '?')}, "
                    f"близость {float(s.get('score', 0.0)):.2f}"
                    for s in mode.rag_sources
                )
            else:
                lines.append("Материалы не использованы (ни один фрагмент не прошёл порог).")
        if mode_key != MODE_OFF:
            lines += _citation_detail_lines(mode)
        if q.expect_abstain:
            continue
        if mode.verdict:
            pairs = zip(mode.verdict.present, q.facts, strict=False)
            marks = "\n".join(
                f"{'✅' if ok else '❌'} {i}. {fact}" for i, (ok, fact) in enumerate(pairs, start=1)
            )
            lines.append(f"Оценка: {_pct(mode.verdict.coverage)} фактов.\n{marks}")
            if mode.verdict.contradicts:
                lines.append("⚠️ Есть утверждение, противоречащее ожиданию.")
            if mode.verdict.note:
                lines.append(f"Замечание оценщика: {mode.verdict.note}")
        else:
            lines.append(f"Оценка не получена: {mode.judge_error or 'нет данных'}")
    return "\n".join(lines)


def _citation_detail_lines(mode: ModeResult) -> list[str]:
    """Детали вопроса про режим «не знаю», цитаты и оценку их соответствия ответу."""
    if mode.abstained:
        return ["🛑 Бот ответил режимом «не знаю» (без обращения к модели)."]
    lines: list[str] = []
    if mode.citations:
        lines.append(
            f"Проверенные цитаты: {len(mode.citations)} (написано моделью: "
            f"{mode.citations_written}):"
        )
        lines.extend(
            f"• {c.get('title', '?')} — фрагмент {c.get('chunk_index', '?')}: "
            f"«{c.get('quote', '')}»"
            for c in mode.citations
        )
    elif mode.citations_unverified:
        lines.append(f"⚠️ Цитаты не подтверждены (написано моделью: {mode.citations_written}).")
    if mode.citation_verdict:
        label = CITATION_VERDICT_LABELS.get(mode.citation_verdict.verdict, "?")
        lines.append(f"Опора ответа на материалы: {label}.")
        lines.extend(
            f"— без опоры в цитатах: {claim}" for claim in mode.citation_verdict.unsupported
        )
    elif mode.citation_judge_error:
        lines.append(f"Оценка цитат не получена: {mode.citation_judge_error}")
    return lines


def split_text(text: str, limit: int) -> list[str]:
    """Режет текст на части не длиннее `limit`, по возможности по границам строк; слишком
    длинная строка режется по символам, ничего не теряется."""
    parts: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) > limit:
            parts.append(current)
            current = line
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts
