"""Чистые правила и тексты сравнения моделей на контрольных вопросах с включённым слоем rag
(команда /research_rag_models, design.md изменения add-rag-models-compare).

Отделён от research/rag_models.py по принципу проекта (как rag_compare_eval.py): здесь ПРАВИЛА и
ТЕКСТЫ — разбор настроек «провайдер:модель», выбор вопросов прогона, метрики обращения (токены в
секунду, холодный старт), агрегаты по моделям (качество, скорость, стабильность), правило
автоостановки, структура отчёта и тексты сводки и деталей вопроса; а обращения к SmartAgent,
судье, фоновый прогон, диск и Telegram — там. Модуль не импортирует ни openai, ни config, ни
Telegram, поэтому все ветвления проверяются тестами (tests/test_rag_models_eval.py) без сети.

Правила оценки ответов (покрытие фактов, цитаты, «не знаю») и контрольный набор — те же, что у
/research_rag_compare, и берутся из rag_compare_eval.py без копирования.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from main_client_settings import default_main_model

from . import rag_compare_eval as ev

PROVIDERS = ("ollama", "deepseek", "kimi")
_PROVIDER_LABELS = {"ollama": "Ollama (локально)", "deepseek": "DeepSeek", "kimi": "Kimi"}
_KEY_ENV_VARS = {"deepseek": "DEEPSEEK_API_KEY", "kimi": "KIMI_API_KEY"}

KIND_MODELS = "models"
REPORT_VERSION = 1
MIN_MODELS = 2
MIN_QUESTIONS = 1
MAX_QUESTIONS = 5

_REASON_MAX_CHARS = 120
_ANSWER_ERROR_TOP = 3


class ModelSettingError(ValueError):
    """Настройка сравнения моделей недопустима — команда отвечает сообщением и не начинает
    прогон."""


# --------------------------------------------------------------------------- #
# Модели сравнения и судья
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """Пара «провайдер:модель» сравнения (или судьи). `model` уже с подставленным умолчанием
    облачного провайдера."""

    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def api_key_env_var(self) -> str | None:
        """Переменная ключа облачного провайдера; у ollama ключа нет."""
        return _KEY_ENV_VARS.get(self.provider)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "provider": self.provider,
            "model": self.model,
            "label": self.label,
        }


def _parse_pair(item: str, defaults: dict[str, str], where: str) -> ModelSpec:
    """Одна пара. Делится по ПЕРВОМУ двоеточию: в имени локальной модели бывают двоеточия
    (`ollama:gpt-oss:20b`)."""
    provider, _, model = item.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in PROVIDERS:
        raise ModelSettingError(
            f"{where}: недопустимый провайдер {provider!r} в {item!r}. "
            f"Допустимо: {', '.join(PROVIDERS)}."
        )
    if provider == "ollama":
        if not model:
            raise ModelSettingError(
                f"{where}: для провайдера ollama модель нужно указать явно "
                "(например, ollama:gpt-oss:20b)."
            )
    elif not model:
        model = defaults.get(provider) or default_main_model(provider)
    return ModelSpec(provider, model)


def parse_model_specs(raw: str, defaults: dict[str, str] | None = None) -> list[ModelSpec]:
    """Список сравниваемых моделей из настройки (пары через запятую). Не менее двух моделей,
    дубликаты запрещены, для `ollama` модель обязательна, у облачных пустая модель — умолчание
    провайдера (`defaults` или основной набор проекта). Нарушение — ModelSettingError с
    пояснением."""
    defaults = defaults or {}
    items = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not items:
        raise ModelSettingError(
            "сравнение моделей не настроено: задайте RAG_MODELS_COMPARE — пары "
            "«провайдер:модель» через запятую (например, ollama:gpt-oss:20b,deepseek:)."
        )
    specs = [_parse_pair(item, defaults, "RAG_MODELS_COMPARE") for item in items]
    if len(specs) < MIN_MODELS:
        raise ModelSettingError(
            f"RAG_MODELS_COMPARE: нужно не меньше {MIN_MODELS} моделей, указана одна."
        )
    seen: set[str] = set()
    for spec in specs:
        if spec.key in seen:
            raise ModelSettingError(f"RAG_MODELS_COMPARE: модель {spec.key} указана дважды.")
        seen.add(spec.key)
    return specs


def parse_judge_spec(raw: str, defaults: dict[str, str] | None = None) -> ModelSpec:
    """Модель-судья из настройки (одна пара «провайдер:модель»)."""
    raw = (raw or "").strip()
    if not raw:
        raise ModelSettingError(
            "судья не настроен: задайте RAG_MODELS_JUDGE — пару «провайдер:модель» "
            "(например, kimi:kimi-k3)."
        )
    if "," in raw:
        raise ModelSettingError("RAG_MODELS_JUDGE: нужна одна пара «провайдер:модель».")
    return _parse_pair(raw, defaults or {}, "RAG_MODELS_JUDGE")


def missing_api_keys(specs: list[ModelSpec], keys: dict[str, bool]) -> list[str]:
    """Названия переменных ключей облачных провайдеров, которые нужны выбранным парам и судье,
    но не заданы (`keys` — провайдер → ключ задан). Без повторов, в порядке появления."""
    missing: list[str] = []
    for spec in specs:
        env_var = spec.api_key_env_var
        if env_var and not keys.get(spec.provider) and env_var not in missing:
            missing.append(env_var)
    return missing


# --------------------------------------------------------------------------- #
# Выбор вопросов прогона
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuestionSelection:
    """Вопросы прогона и пояснения для чата. `error` — прогон начинать нельзя."""

    questions: tuple[ev.Question, ...] = ()
    notes: tuple[str, ...] = ()
    error: str | None = None


def select_questions(questions: list[ev.Question], count: int) -> QuestionSelection:
    """Первые `count` самостоятельных вопросов по корпусу (порядок набора) плюс один вопрос вне
    корпуса (первый `expect_abstain`), если он есть. Многоходовые (с `history`) не участвуют.
    Вопросов по корпусу меньше `count` — прогон по имеющимся с пометкой; нет совсем — ошибка."""
    corpus = [q for q in questions if not q.expect_abstain and not q.history]
    abstain = [q for q in questions if q.expect_abstain and not q.history]
    multi_turn = [q for q in questions if q.history]
    notes: list[str] = []
    if multi_turn:
        notes.append(
            f"Многоходовые вопросы (с историей диалога) пропущены: {len(multi_turn)} — "
            + ", ".join(str(q.number) for q in multi_turn)
            + "."
        )
    if not corpus:
        return QuestionSelection(
            notes=tuple(notes),
            error=(
                "в наборе нет самостоятельных вопросов по корпусу (вопросы вне корпуса и "
                "многоходовые не участвуют). Добавьте вопросы в файл RAG_COMPARE_QUESTIONS_FILE."
            ),
        )
    chosen = corpus[:count]
    if len(chosen) < count:
        notes.append(
            f"В наборе вопросов по корпусу меньше настроенных {count}: прогон пойдёт по "
            f"имеющимся ({len(chosen)})."
        )
    if abstain:
        chosen = chosen + abstain[:1]
    else:
        notes.append("В наборе нет вопроса вне корпуса: проверка отказа «не знаю» не выполняется.")
    return QuestionSelection(questions=tuple(chosen), notes=tuple(notes))


def question_budget_seconds(base_seconds: float, models: int, repeats: int) -> float:
    """Бюджет времени на вопрос: базовые секунды `rag_compare` рассчитаны на два ответа, здесь
    ответов `models × repeats` (каждый ещё и оценивается)."""
    return base_seconds * models * repeats / 2


# --------------------------------------------------------------------------- #
# Метрики обращения
# --------------------------------------------------------------------------- #


def tokens_per_second(completion_tokens: int | None, elapsed: float) -> float | None:
    """Токены ответа в секунду. None — провайдер не вернул токены или время нулевое: скорость
    неизвестна и в агрегаты не входит."""
    if not completion_tokens or completion_tokens <= 0 or elapsed <= 0:
        return None
    return completion_tokens / elapsed


@dataclass
class Attempt:
    """Одно обращение: повтор `repeat` (с единицы) модели `model_key` по вопросу. `cold` — первое
    обращение к модели в прогоне (холодный старт)."""

    model_key: str
    repeat: int
    result: ev.ModeResult
    cold: bool = False

    @property
    def tokens_per_second(self) -> float | None:
        return tokens_per_second(self.result.completion_tokens, self.result.elapsed)

    @property
    def answered(self) -> bool:
        """Ответ получен (не сбой и не превышение предела времени)."""
        return self.result.answer is not None and not self.result.error

    def to_dict(self) -> dict:
        data = self.result.to_dict()
        data["repeat"] = self.repeat
        data["cold"] = self.cold
        data["tokens_per_second"] = self.tokens_per_second
        return data

    @staticmethod
    def from_dict(model_key: str, data: dict) -> Attempt:
        return Attempt(
            model_key=model_key,
            repeat=int(data.get("repeat", 0)),
            result=ev.ModeResult.from_dict(data),
            cold=bool(data.get("cold", False)),
        )


@dataclass
class QuestionRun:
    """Итог вопроса: общий поиск и обращения всех моделей. `search_error` — причина сбоя поиска:
    вопрос исключён из сравнения, обращений к моделям нет."""

    question: ev.Question
    search_sources: list[dict] = field(default_factory=list)
    search_seconds: float = 0.0
    search_error: str | None = None
    attempts: dict[str, list[Attempt]] = field(default_factory=dict)

    @property
    def excluded(self) -> bool:
        return self.search_error is not None

    def all_attempts(self) -> list[Attempt]:
        return [a for items in self.attempts.values() for a in items]

    def to_dict(self) -> dict:
        q = self.question
        return {
            "number": q.number,
            "question": q.question,
            "kind": q.kind,
            "facts": list(q.facts),
            "sources": list(q.sources),
            "expect_abstain": q.expect_abstain,
            "search": {
                "sources": list(self.search_sources),
                "seconds": self.search_seconds,
                "failed": self.search_error,
            },
            "attempts": {
                key: [a.to_dict() for a in items] for key, items in self.attempts.items()
            },
        }

    @staticmethod
    def from_dict(data: dict) -> QuestionRun:
        question = ev.Question(
            number=int(data["number"]),
            question=data["question"],
            kind=data["kind"],
            facts=tuple(data.get("facts", ())),
            sources=tuple(data.get("sources", ())),
            expect_abstain=bool(data.get("expect_abstain", False)),
        )
        search = data.get("search") or {}
        return QuestionRun(
            question=question,
            search_sources=list(search.get("sources", [])),
            search_seconds=float(search.get("seconds", 0.0)),
            search_error=search.get("failed"),
            attempts={
                key: [Attempt.from_dict(key, item) for item in items]
                for key, items in (data.get("attempts") or {}).items()
            },
        )


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _REASON_MAX_CHARS else text[: _REASON_MAX_CHARS - 1] + "…"


# --------------------------------------------------------------------------- #
# Остановка прогона
# --------------------------------------------------------------------------- #


def question_all_failed(run: QuestionRun) -> bool:
    """Вопрос «провалился» для автоостановки: поиск не удался (сервер эмбеддингов недоступен —
    дальше идти бессмысленно) либо ВСЕ его обращения завершились сбоем ответа. Сбой одной модели
    — не повод прерывать вторую; сбой судьи сбоем генерации не считается."""
    if run.search_error is not None:
        return True
    attempts = run.all_attempts()
    return bool(attempts) and all(not a.answered for a in attempts)


def consecutive_failures(runs: list[QuestionRun]) -> int:
    count = 0
    for run in reversed(runs):
        if not question_all_failed(run):
            break
        count += 1
    return count


def should_stop(runs: list[QuestionRun], limit: int) -> bool:
    """Пора ли остановить прогон: подряд не меньше `limit` проваленных вопросов."""
    return limit > 0 and consecutive_failures(runs) >= limit


# --------------------------------------------------------------------------- #
# Агрегаты по моделям
# --------------------------------------------------------------------------- #


def question_numbers(report: dict) -> list[int]:
    """Номера вопросов набора, обработанных прогоном (по ним работает команда деталей)."""
    return [q["number"] for q in report.get("questions", []) if isinstance(q.get("number"), int)]


def _graded(run: QuestionRun) -> bool:
    return not run.excluded


def _citation_eligible(run: QuestionRun, attempt: Attempt) -> bool:
    """Участвует ли ответ в метриках цитат: вопрос по корпусу, ответ получен и не «не знаю»."""
    return (
        not run.question.expect_abstain
        and attempt.answered
        and not attempt.result.search_failed
        and not attempt.result.abstained
    )


def aggregate_model(runs: list[QuestionRun], model_key: str) -> dict:
    """Метрики одной модели по вопросам без сбоя поиска (такие вопросы исключаются из ВСЕХ
    метрик и счётчиков). Качество — только по ответам, оценённым судьёй; сбой судьи — отдельно
    от сбоя генерации; скорость — по всем обращениям с токенами, кроме холодного."""
    included = [r for r in runs if _graded(r)]
    pairs = [(r, a) for r in included for a in r.attempts.get(model_key, [])]

    answered = [(r, a) for r, a in pairs if a.answered]
    failures = Counter(_short(a.result.error) for _, a in pairs if a.result.error)

    speeds = [
        a.tokens_per_second for _, a in answered if not a.cold and a.tokens_per_second is not None
    ]
    cold = [a for _, a in pairs if a.cold]
    cold_answered = [a for a in cold if a.answered]
    # Ответ «не знаю» без вызова модели (llm_calls == 0) токенов не имеет по построению — это не
    # «провайдер не вернул токены».
    speed_unknown = sum(
        1
        for _, a in answered
        if not a.cold and a.tokens_per_second is None and a.result.llm_calls > 0
    )

    fact_pairs = [(r, a) for r, a in answered if not r.question.expect_abstain]
    judged = [(r, a) for r, a in fact_pairs if a.result.verdict is not None]
    unjudged = [(r, a) for r, a in fact_pairs if a.result.verdict is None]
    coverages = [a.result.verdict.coverage for _, a in judged]

    eligible = [(r, a) for r, a in answered if _citation_eligible(r, a)]
    with_quotes = sum(1 for _, a in eligible if a.result.citations)
    unverified = sum(1 for _, a in eligible if a.result.citations_unverified)
    cit_verdicts = {key: 0 for key in ev.CITATION_VERDICTS}
    for _, a in eligible:
        verdict = a.result.citation_verdict
        if verdict and verdict.verdict in cit_verdicts:
            cit_verdicts[verdict.verdict] += 1

    abstain_pairs = [(r, a) for r, a in answered if r.question.expect_abstain]

    spread: dict[int, tuple[float, float, int]] = {}
    for r in included:
        values = [
            a.result.verdict.coverage
            for a in r.attempts.get(model_key, [])
            if a.answered and a.result.verdict is not None
        ]
        if values and not r.question.expect_abstain:
            spread[r.question.number] = (min(values), max(values), len(r.question.facts))

    return {
        "model_key": model_key,
        "attempts": len(pairs),
        "answered": len(answered),
        "failures": dict(failures),
        "speed_n": len(speeds),
        "speed_median": statistics.median(speeds) if speeds else None,
        "speed_min": min(speeds) if speeds else None,
        "speed_max": max(speeds) if speeds else None,
        "speed_unknown": speed_unknown,
        "cold_count": len(cold),
        "cold_tps": cold_answered[0].tokens_per_second if cold_answered else None,
        "cold_elapsed": cold_answered[0].result.elapsed if cold_answered else None,
        "cold_failed": len(cold) - len(cold_answered),
        "fact_answers": len(fact_pairs),
        "judged": len(judged),
        "unjudged": len(unjudged),
        "judge_errors": Counter(
            _short(a.result.judge_error) for _, a in unjudged if a.result.judge_error
        ),
        "avg_coverage": sum(coverages) / len(coverages) if coverages else None,
        "contradicts": sum(1 for _, a in judged if a.result.verdict.contradicts),
        "cit_eligible": len(eligible),
        "cit_quotes": with_quotes,
        "cit_unverified": unverified,
        "cit_verdicts": cit_verdicts,
        # Ответы с проверенными цитатами, для которых судья цитат не вернул оценку (сбой судьи).
        "cit_judge_missing": sum(
            1 for _, a in eligible if a.result.citations and a.result.citation_verdict is None
        ),
        "false_abstain": sum(1 for _, a in fact_pairs if a.result.abstained),
        "abstain_answers": len(abstain_pairs),
        "abstain_correct": sum(1 for _, a in abstain_pairs if a.result.abstained),
        # Среднее число вызовов модели считается по ответам, где модель вызывалась (ответ «не
        # знаю» без вызова в среднее не входит).
        "llm_calls_avg": (
            sum(a.result.llm_calls for _, a in answered if a.result.llm_calls > 0)
            / sum(1 for _, a in answered if a.result.llm_calls > 0)
            if any(a.result.llm_calls > 0 for _, a in answered)
            else None
        ),
        "spread": spread,
    }


def excluded_questions(runs: list[QuestionRun]) -> list[dict]:
    """Вопросы, исключённые из сравнения (сбой поиска), с причиной."""
    return [
        {"number": r.question.number, "reason": r.search_error or ev.SEARCH_FAILED_REASON}
        for r in runs
        if r.excluded
    ]


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #

_MODELS_REPORT_SUFFIX = "_models.json"
_MODELS_REPORT_NAME = re.compile(r"^\d{8}T\d{6}Z_models\.json$")


def report_file_name(stamp: str) -> str:
    """Имя файла отчёта сравнения моделей по метке времени `YYYYmmddTHHMMSSZ`. Не совпадает с
    шаблоном отчётов /research_rag_compare, поэтому те команды его не видят."""
    return stamp + _MODELS_REPORT_SUFFIX


def latest_models_report_name(names: list[str]) -> str | None:
    """Имя последнего отчёта сравнения моделей среди файлов каталога; None — отчётов нет."""
    reports = sorted(name for name in names if _MODELS_REPORT_NAME.match(name))
    return reports[-1] if reports else None


def is_models_report(report: Any) -> bool:
    """Отчёт — сравнения моделей текущего формата с нужными полями. Отчёты другого вида и других
    версий не читаются как отчёты моделей."""
    return (
        isinstance(report, dict)
        and report.get("kind") == KIND_MODELS
        and report.get("version") == REPORT_VERSION
        and isinstance(report.get("questions"), list)
        and isinstance(report.get("models"), list)
        and isinstance(report.get("settings"), dict)
    )


def build_report(
    *,
    started_at: str,
    finished_at: str,
    settings: dict,
    models: list[ModelSpec],
    judge: ModelSpec,
    runs: list[QuestionRun],
    planned: int,
    repeats: int,
    status: str = ev.STATUS_COMPLETED,
    stop_reason: str | None = None,
) -> dict:
    """Отчёт прогона (его пишет на диск rag_models.py): настройки, модели, судья и по каждому
    вопросу — поиск и ВСЕ повторы всех моделей. Тексты цитируемых фрагментов в отчёт не пишутся."""
    return {
        "kind": KIND_MODELS,
        "version": REPORT_VERSION,
        "status": status,
        "stop_reason": stop_reason,
        "planned": planned,
        "repeats": repeats,
        "started_at": started_at,
        "finished_at": finished_at,
        "settings": settings,
        "models": [m.to_dict() for m in models],
        "judge": judge.to_dict(),
        "questions": [r.to_dict() for r in runs],
    }


# --------------------------------------------------------------------------- #
# Тексты (обычный текст без разметки Telegram)
# --------------------------------------------------------------------------- #


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{round(value * 100)}%"


def _tps(value: float | None) -> str:
    return "неизвестна" if value is None else f"{value:.1f} ток/с"


def _share(part: int, whole: int) -> str:
    return f"{part} из {whole}"


def judge_data_line(judge: dict) -> str:
    """Строка о передаче данных судье (прозрачность канала)."""
    provider = judge.get("provider", "?")
    label = judge.get("label", provider)
    if provider == "ollama":
        return (
            f"Судья — {label} (локальный сервер Ollama): вопросы, ответы и цитируемые "
            "фрагменты передавались ему, за пределы вашего сервера не уходили."
        )
    name = _PROVIDER_LABELS.get(provider, provider)
    return (
        f"⚠️ Вопросы, ответы и тексты цитируемых фрагментов корпуса передавались судье — "
        f"облачному провайдеру {name} ({label})."
    )


def _settings_line(settings: dict) -> str:
    parts = [
        f"стратегия {settings.get('strategy', '?')}",
        f"top-k {settings.get('top_k', '?')}",
        f"порог {settings.get('min_score', '?')}",
    ]
    if settings.get("candidates"):
        parts.append(f"кандидатов {settings['candidates']}")
    parts.append("переписывание вопроса выключено")
    return "Настройки поиска: " + ", ".join(parts) + "."


def _spread_text(spread: dict[int, tuple[float, float, int]]) -> str:
    items = []
    for number in sorted(spread):
        low, high, facts = spread[number]
        low_n, high_n = round(low * facts), round(high * facts)
        items.append(
            f"вопрос {number}: {low_n}–{high_n} из {facts}" if low_n != high_n
            else f"вопрос {number}: {high_n} из {facts}"
        )
    return "; ".join(items)


def _model_block(index: int, model: dict, agg: dict) -> list[str]:
    lines = [f"\n{index}. {model.get('label', agg['model_key'])}"]
    # Качество.
    if agg["judged"]:
        quality = [
            f"покрытие фактов {_pct(agg['avg_coverage'])} (оценено ответов: {agg['judged']})"
        ]
    else:
        quality = ["нет ответов, оценённых судьёй"]
    if agg["judged"]:
        quality.append(f"противоречий: {agg['contradicts']}")
    lines.append("   Качество: " + ", ".join(quality) + ".")
    if agg["cit_eligible"]:
        verdicts = agg["cit_verdicts"]
        lines.append(
            f"   Цитаты: проверенные в {_share(agg['cit_quotes'], agg['cit_eligible'])} ответов; "
            f"«цитаты не подтверждены» — {_share(agg['cit_unverified'], agg['cit_eligible'])}; "
            "опора на материалы: "
            + ", ".join(f"{ev.CITATION_VERDICT_LABELS[k]} — {verdicts[k]}" for k in verdicts)
            + (
                f"; без оценки судьи — {agg['cit_judge_missing']}"
                if agg["cit_judge_missing"]
                else ""
            )
            + "."
        )
    if agg["abstain_answers"]:
        lines.append(
            f"   Вопрос вне корпуса: верный отказ в "
            f"{_share(agg['abstain_correct'], agg['abstain_answers'])} ответов."
        )
    if agg["false_abstain"]:
        lines.append(f"   ⚠️ Ложных отказов («не знаю») по корпусу: {agg['false_abstain']}.")
    # Скорость.
    if agg["speed_n"]:
        lines.append(
            f"   Скорость: медиана {_tps(agg['speed_median'])}, мин {_tps(agg['speed_min'])}, "
            f"макс {_tps(agg['speed_max'])} (обращений: {agg['speed_n']})."
        )
    else:
        lines.append(
            "   Скорость: нет данных вне холодного старта (мало обращений или токены неизвестны)."
            if agg["cold_tps"] is not None
            else "   Скорость: нет данных (токены ответа неизвестны или нет ответов)."
        )
    if agg["speed_unknown"]:
        lines.append(
            f"   Скорость неизвестна (провайдер не вернул токены): {agg['speed_unknown']}."
        )
    if agg["cold_count"]:
        if agg["cold_tps"] is not None:
            lines.append(
                f"   Холодный старт (первое обращение, не входит в статистику): "
                f"{_tps(agg['cold_tps'])}, {agg['cold_elapsed']:.0f} с."
            )
        else:
            lines.append("   Холодный старт: первое обращение не завершилось ответом.")
    # Стабильность.
    lines.append(
        f"   Стабильность: ответ получен в {_share(agg['answered'], agg['attempts'])} обращений."
    )
    if agg["failures"]:
        top = Counter(agg["failures"]).most_common(_ANSWER_ERROR_TOP)
        lines.append("   Причины сбоев: " + "; ".join(f"{reason} ×{n}" for reason, n in top) + ".")
    if agg["unjudged"]:
        reasons = "; ".join(f"{r} ×{n}" for r, n in agg["judge_errors"].most_common(2))
        lines.append(
            f"   Не оценено судьёй: {agg['unjudged']}" + (f" ({reasons})" if reasons else "") + "."
        )
    if agg["llm_calls_avg"] is not None:
        lines.append(f"   Вызовов модели на ответ в среднем: {agg['llm_calls_avg']:.1f}.")
    if agg["spread"]:
        lines.append("   Разброс покрытия между повторами — " + _spread_text(agg["spread"]) + ".")
    return lines


def format_summary(report: dict) -> str:
    """Сводка отчёта сравнения моделей: настройки, число вопросов и повторов, по каждой модели —
    качество, скорость и стабильность. «Победитель» не объявляется."""
    runs = [QuestionRun.from_dict(q) for q in report.get("questions", [])]
    models = report.get("models", [])
    settings = report.get("settings", {})
    repeats = report.get("repeats", settings.get("repeats", "?"))
    corpus = sum(1 for r in runs if not r.question.expect_abstain and not r.excluded)
    abstain = sum(1 for r in runs if r.question.expect_abstain and not r.excluded)

    lines = ["📊 Сравнение моделей с RAG: качество, скорость, стабильность"]
    if report.get("status") == ev.STATUS_STOPPED:
        planned = report.get("planned", len(runs))
        reason = report.get("stop_reason") or "причина не указана"
        lines.append(
            f"⛔ Прогон остановлен: {reason}. Обработано {len(runs)} из {planned} вопросов; "
            "метрики ниже — только по обработанным."
        )
    if report.get("finished_at"):
        verb = "остановлен" if report.get("status") == ev.STATUS_STOPPED else "завершён"
        lines.append(f"Прогон {verb}: {report['finished_at']} (UTC).")
    judge = report.get("judge") or {}
    lines.append("Модели: " + "; ".join(m.get("label", "?") for m in models) + ".")
    temperature = (
        "температуру задаёт сам Kimi" if judge.get("provider") == "kimi" else "temperature=0"
    )
    lines.append(f"Судья: {judge.get('label', '?')} (оценка вслепую; {temperature}).")
    lines.append(_settings_line(settings))
    lines.append(
        f"Выборка: вопросов по корпусу — {corpus}, вне корпуса — {abstain}, повторов каждой "
        f"модели на вопрос — {repeats}. Выборка малая: цифры — ориентир, не статистика."
        + (
            f" Не вошли в сравнение из-за сбоя поиска: {sum(1 for r in runs if r.excluded)}."
            if any(r.excluded for r in runs)
            else ""
        )
    )
    for index, model in enumerate(models, start=1):
        lines += _model_block(index, model, aggregate_model(runs, model.get("key", "")))

    excluded = excluded_questions(runs)
    if excluded:
        details = "; ".join(f"{e['number']} ({_short(e['reason'])})" for e in excluded)
        lines.append(
            f"\nИсключено из сравнения (поиск не удался, к моделям не обращались): "
            f"{len(excluded)} — {details}."
        )
    search_times = [r.search_seconds for r in runs if not r.excluded and r.search_seconds]
    if search_times:
        lines.append(
            "Поиск по индексу: один на вопрос, в среднем "
            f"{sum(search_times) / len(search_times):.1f} "
            "с; в скорость не входит, все модели получили одинаковые фрагменты."
        )
    lines.append(
        "\nСкорость = токены ответа / время ответа (все вызовы модели в ответе, повтор за "
        "цитатами включён, поиск нет). Токены рассуждений входят в токены ответа у проверенных "
        "провайдеров; токенизаторы моделей различаются, поэтому сравнение скорости приблизительно."
    )
    if judge:
        lines.append(judge_data_line(judge))
    lines.append(
        "Выводы о лучшей модели делает читающий: бот победителя не объявляет.\n"
        "Детали вопроса: /research_rag_models_report <номер> (номера из набора: "
        + (", ".join(str(n) for n in question_numbers(report)) or "нет")
        + ")."
    )
    return "\n".join(lines)


def _attempt_lines(label: str, attempt: Attempt, question: ev.Question) -> list[str]:
    result = attempt.result
    title = f"— {label}, повтор {attempt.repeat}" + (" (холодный старт)" if attempt.cold else "")
    lines = [f"\n{title} —"]
    if not attempt.answered:
        lines.append(f"Сбой ответа: {result.error or 'ответ не получен'}")
        return lines
    lines.append(result.answer or "(пустой ответ)")
    tokens = (
        f"{result.completion_tokens} токенов ответа"
        if result.completion_tokens is not None
        else "токены ответа неизвестны"
    )
    prompt = f", промпт {result.prompt_tokens}" if result.prompt_tokens is not None else ""
    lines.append(
        f"Скорость: {_tps(attempt.tokens_per_second)} · {tokens}{prompt} · "
        f"{result.llm_calls} выз. модели · {result.elapsed:.1f} с"
    )
    lines += ev._citation_detail_lines(result)  # noqa: SLF001 — общий формат деталей цитат
    if question.expect_abstain:
        lines.append("Отказ «не знаю»: " + ("да ✓" if result.abstained else "нет ✗ (дан ответ)"))
        return lines
    if result.verdict:
        pairs = zip(result.verdict.present, question.facts, strict=False)
        marks = "\n".join(
            f"{'✅' if ok else '❌'} {i}. {fact}" for i, (ok, fact) in enumerate(pairs, start=1)
        )
        lines.append(f"Оценка: {_pct(result.verdict.coverage)} фактов.\n{marks}")
        if result.verdict.contradicts:
            lines.append("⚠️ Есть утверждение, противоречащее ожиданию.")
        if result.verdict.note:
            lines.append(f"Замечание судьи: {result.verdict.note}")
    else:
        lines.append(f"Оценка не получена: {result.judge_error or 'нет данных'}")
    return lines


def format_question_detail(report: dict, number: int) -> str | None:
    """Детали вопроса: ответы всех моделей по всем повторам рядом (по повторам, внутри — по
    моделям) с вердиктами, цитатами, скоростью и токенами. None — вопроса с таким номером нет."""
    for raw in report.get("questions", []):
        if raw.get("number") == number:
            run = QuestionRun.from_dict(raw)
            break
    else:
        return None
    q = run.question
    lines = [f"❓ Вопрос {q.number} ({ev.KIND_LABELS.get(q.kind, q.kind)}): {q.question}"]
    if q.expect_abstain:
        lines.append("\nОжидание: ответ «не знаю» (вопрос вне корпуса).")
    else:
        lines += [
            "\nОжидаемые факты:",
            *[f"{i}. {fact}" for i, fact in enumerate(q.facts, start=1)],
            "\nОжидаемые источники: " + "; ".join(q.sources),
        ]
    if run.excluded:
        lines.append(
            f"\n⚠️ Вопрос исключён из сравнения: {run.search_error}. "
            "К моделям по нему не обращались."
        )
        return "\n".join(lines)
    if run.search_sources:
        lines.append(f"\nОбщий поиск (один на вопрос, {run.search_seconds:.1f} с), фрагменты:")
        lines.extend(
            f"• {s.get('title', '?')} — фрагмент {s.get('chunk_index', '?')}, "
            f"близость {float(s.get('score', 0.0)):.2f}"
            for s in run.search_sources
        )
    else:
        lines.append("\nОбщий поиск: ни один фрагмент не прошёл порог.")

    models = report.get("models", [])
    repeats = sorted({a.repeat for a in run.all_attempts()})
    for repeat in repeats:
        for model in models:
            for attempt in run.attempts.get(model.get("key", ""), []):
                if attempt.repeat == repeat:
                    lines += _attempt_lines(model.get("label", "?"), attempt, q)
    return "\n".join(lines)
