"""Чистые правила оснастки подбора параметров локальной модели Ollama (`scripts/ollama_tune.py`,
спека `openspec/changes/add-ollama-tuning-script/`): разбор файла вариантов, агрегация метрик,
таблица «до/после» и тексты. Модуль не обращается ни к сети, ни к LLM, ни к диску, поэтому все
ветвления проверяются тестами (`tests/test_ollama_tune_lib.py`).

Модель данных отчёта переиспользует `research/rag_models_eval.py` (`QuestionRun`, `Attempt`,
`aggregate_model`): вариант настроек играет роль «модели» (ключ — имя варианта), а к каждому
обращению добавляется поле `calls` — записанные обёрткой клиента `usage` и `finish_reason`.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from research import rag_compare_eval as ev
from research import rag_models_eval as rme

KIND = "ollama_tune"
REPORT_VERSION = 1

# Временные теги Ollama, которые создаёт скрипт: общий префикс позволяет убрать остатки
# прерванного прогона (design.md, раздел Risks).
TAG_PREFIX = "tunetmp-"

# Допустимые параметры Modelfile варианта и их тип. Список закрытый: опечатка в имени параметра
# молча ушла бы в Ollama и «улучшение» ничего бы не меняло.
PARAM_TYPES: dict[str, type] = {
    "num_ctx": int,
    "num_thread": int,
    "num_batch": int,
    "num_gpu": int,
    "top_k": int,
    "seed": int,
    "temperature": float,
    "top_p": float,
    "min_p": float,
    "repeat_penalty": float,
}
_POSITIVE_PARAMS = {"num_ctx", "num_thread", "num_batch"}

# Бот всегда шлёт `max_tokens` (MAX_OUTPUT_TOKENS), а он перекрывает `num_predict` Modelfile
# (проверено, design.md, Results of Spikes) — лимит ответа варианта задаётся полем
# `max_output_tokens`, которое переносится в `.env`.
_OVERRIDDEN_PARAMS = {"num_predict": "max_output_tokens"}

# `none` не выключает рассуждение (gpt-oss) или выводит его в видимый ответ (qwen3) — см. design.md,
# раздел Results of Spikes.
REASONING_EFFORTS = ("low", "medium", "high")

MIN_VARIANTS = 2

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
_REASON_MAX_CHARS = 120


class VariantFileError(ValueError):
    """Файл вариантов недопустим: скрипт сообщает причину и не обращается к Ollama и моделям."""


# --------------------------------------------------------------------------- #
# Варианты настроек
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Variant:
    """Один вариант настроек. `params` — параметры Modelfile (временный тег), `reasoning_effort` —
    параметр уровня запроса, `system_prompt` — экспериментальный вариант промпта."""

    name: str
    base: str
    params: dict[str, int | float] = field(default_factory=dict)
    reasoning_effort: str | None = None
    system_prompt: str | None = None
    max_output_tokens: int | None = None
    baseline: bool = False

    @property
    def label(self) -> str:
        return self.name

    @property
    def needs_tag(self) -> bool:
        return bool(self.params)

    @property
    def request_level(self) -> bool:
        return self.reasoning_effort is not None

    @property
    def experimental_prompt(self) -> bool:
        return self.system_prompt is not None

    def describe(self) -> str:
        parts = [f"{key}={value}" for key, value in sorted(self.params.items())]
        if self.max_output_tokens is not None:
            parts.append(f"max_output_tokens={self.max_output_tokens}")
        if self.reasoning_effort:
            parts.append(f"reasoning_effort={self.reasoning_effort}")
        if self.system_prompt is not None:
            parts.append("свой системный промпт")
        return ", ".join(parts) if parts else "настройки по умолчанию"

    def to_dict(self) -> dict:
        return {
            "key": self.name,
            "label": self.label,
            "name": self.name,
            "base": self.base,
            "params": dict(self.params),
            "reasoning_effort": self.reasoning_effort,
            "system_prompt": self.system_prompt,
            "max_output_tokens": self.max_output_tokens,
            "baseline": self.baseline,
            "portability": portability(self),
        }

    @staticmethod
    def from_dict(data: dict) -> Variant:
        return Variant(
            name=data["name"],
            base=data.get("base", ""),
            params=dict(data.get("params") or {}),
            reasoning_effort=data.get("reasoning_effort"),
            system_prompt=data.get("system_prompt"),
            max_output_tokens=data.get("max_output_tokens"),
            baseline=bool(data.get("baseline", False)),
        )


@dataclass(frozen=True)
class VariantSet:
    variants: tuple[Variant, ...]
    environment: str = ""

    @property
    def baseline(self) -> Variant:
        return next(v for v in self.variants if v.baseline)


def _parse_param(variant: str, key: str, value: Any) -> int | float:
    if key in _OVERRIDDEN_PARAMS:
        raise VariantFileError(
            f"вариант {variant!r}: параметр {key} в Modelfile перекрывается max_tokens запроса "
            f"бота и ничего не меняет — задайте лимит ответа полем {_OVERRIDDEN_PARAMS[key]}."
        )
    expected = PARAM_TYPES.get(key)
    if expected is None:
        raise VariantFileError(
            f"вариант {variant!r}: неизвестный параметр {key!r}. "
            f"Допустимо: {', '.join(sorted(PARAM_TYPES))}."
        )
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VariantFileError(f"вариант {variant!r}: параметр {key} должен быть числом.")
    if expected is int:
        if isinstance(value, float) and not value.is_integer():
            raise VariantFileError(f"вариант {variant!r}: параметр {key} должен быть целым.")
        number: int | float = int(value)
    else:
        number = float(value)
    if key in _POSITIVE_PARAMS and number <= 0:
        raise VariantFileError(f"вариант {variant!r}: параметр {key} должен быть больше нуля.")
    return number


def parse_variants(raw: Any) -> VariantSet:
    """Проверяет и разбирает файл вариантов: `{"environment": "...", "variants": [...]}`.
    Нарушение (нет исходного варианта, повтор имён, меньше двух вариантов, неизвестный параметр)
    — VariantFileError."""
    if not isinstance(raw, dict):
        raise VariantFileError("файл вариантов должен быть объектом с полем «variants»")
    items = raw.get("variants")
    if not isinstance(items, list) or not items:
        raise VariantFileError("в файле нет списка «variants» или он пуст")
    environment = raw.get("environment", "")
    if not isinstance(environment, str):
        raise VariantFileError("поле «environment» должно быть строкой")

    variants: list[Variant] = []
    seen: set[str] = set()
    for number, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise VariantFileError(f"вариант {number}: ожидался объект")
        name = item.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise VariantFileError(
                f"вариант {number}: нужно имя из латиницы, цифр и «._-» (до 40 символов)"
            )
        if name in seen:
            raise VariantFileError(f"имя варианта {name!r} повторяется")
        seen.add(name)
        base = item.get("base")
        if not isinstance(base, str) or not base.strip():
            raise VariantFileError(f"вариант {name!r}: не указана базовая модель «base»")
        params_raw = item.get("params", {})
        if not isinstance(params_raw, dict):
            raise VariantFileError(f"вариант {name!r}: «params» должен быть объектом")
        params = {key: _parse_param(name, key, value) for key, value in params_raw.items()}
        effort = item.get("reasoning_effort")
        if effort is not None and effort not in REASONING_EFFORTS:
            raise VariantFileError(
                f"вариант {name!r}: reasoning_effort должен быть одним из "
                f"{', '.join(REASONING_EFFORTS)} (none рассуждение не выключает — см. design.md)."
            )
        prompt = item.get("system_prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise VariantFileError(f"вариант {name!r}: system_prompt должен быть непустой строкой")
        max_tokens = item.get("max_output_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
        ):
            raise VariantFileError(
                f"вариант {name!r}: max_output_tokens должен быть целым больше нуля"
            )
        variants.append(
            Variant(
                name=name,
                base=base.strip(),
                params=params,
                reasoning_effort=effort,
                system_prompt=prompt,
                max_output_tokens=max_tokens,
                baseline=bool(item.get("baseline", False)),
            )
        )

    if len(variants) < MIN_VARIANTS:
        raise VariantFileError(f"нужно не меньше {MIN_VARIANTS} вариантов, указан один")
    baselines = [v.name for v in variants if v.baseline]
    if not baselines:
        raise VariantFileError("не помечен исходный вариант: добавьте «baseline»: true одному")
    if len(baselines) > 1:
        raise VariantFileError("исходным может быть только один вариант: " + ", ".join(baselines))
    return VariantSet(tuple(variants), environment.strip())


def select_variants(variant_set: VariantSet, only: list[str]) -> VariantSet:
    """Оставляет варианты `only` (порядок файла) и всегда исходный: без него нет «до». Неизвестное
    имя — VariantFileError."""
    if not only:
        return variant_set
    known = {v.name for v in variant_set.variants}
    unknown = [name for name in only if name not in known]
    if unknown:
        raise VariantFileError(
            "нет вариантов с именами: " + ", ".join(unknown) + ". В файле: "
            + ", ".join(sorted(known)) + "."
        )
    chosen = tuple(v for v in variant_set.variants if v.baseline or v.name in only)
    if len(chosen) < MIN_VARIANTS:
        raise VariantFileError("после отбора --only остался один вариант: сравнивать не с чем")
    return VariantSet(chosen, variant_set.environment)


def portability(variant: Variant) -> dict:
    """Переносится ли конфигурация варианта в бота без правок кода (спека, «Что переносится»)."""
    notes: list[str] = []
    level = "bot"
    if variant.request_level:
        level = "request"
        notes.append(
            "reasoning_effort задаётся только в запросе: перенос требует доработки бота и "
            "отдельного решения владельца"
        )
    if variant.experimental_prompt:
        notes.append(
            "экспериментальный системный промпт: рабочий SYSTEM_PROMPT скрипт не менял; замена — "
            "ручное решение оператора с сохранением предупреждений и осторожных формулировок"
        )
        if level == "bot":
            level = "prompt"
    if not notes:
        carried = []
        if variant.params:
            carried.append("параметры в Modelfile рабочего тега (MAIN_MODEL)")
        if variant.max_output_tokens is not None:
            carried.append("лимит ответа в MAX_OUTPUT_TOKENS (.env)")
        if carried:
            notes.append("переносится: " + " и ".join(carried))
        else:
            notes.append("переносится: настройки по умолчанию, другого не требуется")
    return {"level": level, "text": "; ".join(notes)}


def temp_tag(run_id: str, name: str) -> str:
    """Имя временного тега варианта: префикс + идентификатор прогона + имя варианта."""
    return f"{TAG_PREFIX}{run_id}-{name}".lower()


def is_temp_tag(model_name: str) -> bool:
    return model_name.lower().startswith(TAG_PREFIX)


def parse_ps_entry(ps: dict, model: str) -> dict | None:
    """Память и контекст модели из ответа `/api/ps`: size, size_vram, context_length. None —
    модели нет среди загруженных."""
    wanted = {model, f"{model}:latest"}
    for item in ps.get("models", []):
        if item.get("name") in wanted or item.get("model") in wanted:
            return {
                "size": item.get("size"),
                "size_vram": item.get("size_vram"),
                "context_length": item.get("context_length"),
            }
    return None


# --------------------------------------------------------------------------- #
# Выбор вопросов
# --------------------------------------------------------------------------- #


def pick_questions(
    questions: list[ev.Question], count: int | None
) -> rme.QuestionSelection:
    """Вопросы прогона: без `count` — все самостоятельные вопросы набора; с `count` — правила
    `/research_rag_models` (первые `count` по корпусу и один вопрос вне корпуса)."""
    if count is not None:
        return rme.select_questions(questions, count)
    independent = [q for q in questions if not q.history]
    skipped = len(questions) - len(independent)
    notes = (
        (f"Многоходовые вопросы пропущены: {skipped}.",) if skipped else ()
    )
    if not independent:
        return rme.QuestionSelection(notes=notes, error="в наборе нет самостоятельных вопросов")
    return rme.QuestionSelection(questions=tuple(independent), notes=notes)


# --------------------------------------------------------------------------- #
# Метрики обращения
# --------------------------------------------------------------------------- #


def call_record(usage: Any, finish_reason: str | None) -> dict:
    """Запись одного обращения к модели для отчёта: токены промпта/ответа и причина завершения."""
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "finish_reason": finish_reason,
    }


def is_truncated(calls: list[dict]) -> bool:
    """Ответ обрезан по лимиту токенов: хотя бы одно обращение завершилось `length`."""
    return any(c.get("finish_reason") == "length" for c in calls)


def attach_calls(question: dict, calls: dict[str, list[list[dict]]]) -> dict:
    """Дополняет словарь вопроса (`QuestionRun.to_dict()`) записями обращений: `calls[имя][i]` —
    обращения i-го повтора варианта в порядке прогона."""
    for name, items in question.get("attempts", {}).items():
        recorded = calls.get(name, [])
        for index, attempt in enumerate(items):
            attempt["calls"] = recorded[index] if index < len(recorded) else []
    return question


# --------------------------------------------------------------------------- #
# Агрегаты по вариантам
# --------------------------------------------------------------------------- #


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate_variant(
    questions: list[dict], name: str, resources: dict | None = None
) -> dict:
    """Метрики одного варианта по вопросам отчёта. Скорость, качество, цитаты и разброс — правила
    `rag_models_eval.aggregate_model` (холодное обращение отдельно, вопросы со сбоем поиска
    исключены); сверху — обрезка по лимиту, токены промпта, загрузка и память."""
    runs = [rme.QuestionRun.from_dict(q) for q in questions]
    agg = rme.aggregate_model(runs, name)

    with_calls = truncated = 0
    prompt_tokens: list[int] = []
    for question in questions:
        if question.get("search", {}).get("failed"):
            continue
        for attempt in question.get("attempts", {}).get(name, []):
            calls = attempt.get("calls") or []
            if calls:
                with_calls += 1
                truncated += is_truncated(calls)
            # Размер промпта — самое большое отдельное обращение (сумма по вызовам завысила бы
            # его у ответов с повтором за цитатами).
            value = max((c.get("prompt_tokens") or 0 for c in calls), default=0) or None
            if value is None:
                value = attempt.get("prompt_tokens")
            if value:
                prompt_tokens.append(value)
    agg["with_calls"] = with_calls
    agg["truncated"] = truncated
    agg["prompt_tokens_median"] = _median([float(p) for p in prompt_tokens])
    agg["prompt_tokens_max"] = max(prompt_tokens) if prompt_tokens else None

    resources = resources or {}
    loads = [s for s in resources.get("load_seconds", []) if s is not None]
    agg["load_median"] = _median(loads)
    agg["load_count"] = len(loads)
    agg["size"] = resources.get("size")
    agg["size_vram"] = resources.get("size_vram")
    agg["context_length"] = resources.get("context_length")
    agg["resource_error"] = resources.get("error")
    return agg


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #

_REPORT_NAME = re.compile(r"^\d{8}T\d{6}Z_tune\.json$")


def report_file_name(stamp: str) -> str:
    return f"{stamp}_tune.json"


def latest_report_name(names: list[str]) -> str | None:
    reports = sorted(name for name in names if _REPORT_NAME.match(name))
    return reports[-1] if reports else None


def is_tune_report(report: Any) -> bool:
    return (
        isinstance(report, dict)
        and report.get("kind") == KIND
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
    variants: list[Variant],
    resources: dict[str, dict],
    questions: list[dict],
    planned: int,
    repeats: int,
    status: str = ev.STATUS_COMPLETED,
    stop_reason: str | None = None,
) -> dict:
    return {
        "kind": KIND,
        "version": REPORT_VERSION,
        "status": status,
        "stop_reason": stop_reason,
        "planned": planned,
        "repeats": repeats,
        "started_at": started_at,
        "finished_at": finished_at,
        "settings": settings,
        # Ключи `key`/`label` — как у моделей `/research_rag_models`: детали вопроса используют
        # тот же формат.
        "models": [v.to_dict() for v in variants],
        "resources": resources,
        "questions": questions,
    }


# --------------------------------------------------------------------------- #
# Тексты
# --------------------------------------------------------------------------- #


def _gib(value: int | float | None) -> str:
    return "—" if not value else f"{value / 1024**3:.1f}"


def _tps_cell(agg: dict) -> str:
    if not agg["speed_n"]:
        return "—"
    median = agg["speed_median"]
    return f"{median:.1f} ({agg['speed_min']:.1f}–{agg['speed_max']:.1f})"


def _delta(value: float | None, base: float | None) -> str:
    if value is None or not base:
        return ""
    change = (value - base) / base * 100
    if abs(change) < 0.5:
        return " (=)"
    return f" ({change:+.0f}%)"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{round(value * 100)}%"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    return [line(headers), "  ".join("-" * w for w in widths), *(line(r) for r in rows)]


def judge_enabled(report: dict) -> bool:
    return bool((report.get("settings") or {}).get("judge"))


def format_summary(report: dict) -> str:
    """Таблица «до/после» по вариантам и пояснения: размер выборки, переносимость, исключённые
    вопросы, канал данных судьи. Победителя не объявляет."""
    questions = report.get("questions", [])
    models = report.get("models", [])
    settings = report.get("settings", {})
    repeats = report.get("repeats", settings.get("repeats", "?"))
    resources = report.get("resources", {})
    judged = judge_enabled(report)
    runs = [rme.QuestionRun.from_dict(q) for q in questions]
    included = [r for r in runs if not r.excluded]

    aggs = {
        m["key"]: aggregate_variant(questions, m["key"], resources.get(m["key"])) for m in models
    }
    base_key = next((m["key"] for m in models if m.get("baseline")), None)
    base = aggs.get(base_key) if base_key else None

    lines = ["Подбор параметров Ollama: «до/после»"]
    if report.get("status") == ev.STATUS_STOPPED:
        lines.append(
            f"⛔ Прогон остановлен: {report.get('stop_reason') or 'причина не указана'}. "
            f"Обработано {len(runs)} из {report.get('planned', len(runs))} вопросов; "
            "цифры — только по обработанным."
        )
    if report.get("finished_at"):
        lines.append(f"Прогон: {report['finished_at']} (UTC).")
    if settings.get("environment"):
        lines.append(f"Окружение сервера Ollama (со слов оператора): {settings['environment']}")
    if settings.get("strategy"):
        search = [
            f"стратегия {settings['strategy']}",
            f"top-k {settings.get('top_k', '?')}",
            f"порог {settings.get('min_score', '?')}",
        ]
        lines.append(
            "Поиск: " + ", ".join(search) + "; переписывание вопроса и рыночные данные выключены."
        )
    lines.append(
        f"Выборка: вопросов — {len(included)}, повторов на вариант — {repeats}. Выборка малая: "
        "цифры — ориентир, не статистика; вывод делает читающий."
    )

    headers = ["Вариант", "Ток/с мед. (мин–макс)", "Загрузка, с", "Память, ГиБ", "Контекст",
               "Промпт, ток.", "Обрезано"]
    if judged:
        headers += ["Покрытие", "Противореч."]
    rows: list[list[str]] = []
    for model in models:
        agg = aggs[model["key"]]
        is_base = model.get("baseline", False)
        name = f"{model['label']} ← до" if is_base else model["label"]
        speed = _tps_cell(agg) + ("" if is_base else _delta(agg["speed_median"],
                                                         base["speed_median"] if base else None))
        load = "—" if agg["load_median"] is None else f"{agg['load_median']:.0f}"
        memory = _gib(agg["size"])
        if agg["size"] and agg["size_vram"]:
            memory += f" (VRAM {_gib(agg['size_vram'])})"
        prompt = "—" if agg["prompt_tokens_max"] is None else (
            f"{agg['prompt_tokens_median']:.0f} (макс {agg['prompt_tokens_max']})"
        )
        trunc = f"{agg['truncated']} из {agg['with_calls']}" if agg["with_calls"] else "—"
        row = [name, speed, load, memory, str(agg["context_length"] or "—"), prompt, trunc]
        if judged:
            row += [
                f"{_pct(agg['avg_coverage'])} ({agg['judged']})" if agg["judged"] else "—",
                str(agg["contradicts"]) if agg["judged"] else "—",
            ]
        rows.append(row)
    lines += ["", *_table(headers, rows)]

    lines.append("")
    for model in models:
        agg = aggs[model["key"]]
        extra: list[str] = []
        if agg["answered"] != agg["attempts"]:
            extra.append(f"ответ получен в {agg['answered']} из {agg['attempts']} обращений")
        if agg["failures"]:
            top = sorted(agg["failures"].items(), key=lambda kv: -kv[1])[:2]
            extra.append("причины сбоев: " + "; ".join(f"{r} ×{n}" for r, n in top))
        if agg["speed_unknown"]:
            extra.append(f"скорость неизвестна у {agg['speed_unknown']} (токены не возвращены)")
        if agg["cold_count"]:
            cold = (
                f"холодное обращение {agg['cold_tps']:.1f} ток/с, {agg['cold_elapsed']:.0f} с"
                if agg["cold_tps"] is not None else "холодное обращение без ответа"
            )
            extra.append(cold + " (в статистику не входит)")
        if judged and agg["unjudged"]:
            extra.append(f"не оценено судьёй: {agg['unjudged']}")
        if judged and agg["abstain_answers"]:
            extra.append(
                f"верный отказ «не знаю»: {agg['abstain_correct']} из {agg['abstain_answers']}"
            )
        if judged and agg["spread"] and isinstance(repeats, int) and repeats > 1:
            extra.append("разброс покрытия между повторами — " + rme._spread_text(agg["spread"]))  # noqa: SLF001
        if agg["resource_error"]:
            extra.append(f"ресурсы не измерены: {agg['resource_error']}")
        describe = Variant.from_dict(model).describe()
        lines.append(f"• {model['label']} [{model['base']}; {describe}]")
        lines.append(f"    {model['portability']['text']}.")
        if extra:
            lines.append("    " + "; ".join(extra) + ".")

    excluded = rme.excluded_questions(runs)
    if excluded:
        details = "; ".join(f"{e['number']} ({rme._short(e['reason'])})" for e in excluded)  # noqa: SLF001
        lines.append(f"\nИсключено из сравнения (поиск не удался): {len(excluded)} — {details}.")
    search_times = [r.search_seconds for r in included if r.search_seconds]
    lines.append(
        "\nСкорость = токены ответа / время обращений к модели (без поиска, без загрузки модели); "
        "холодное обращение отдельно. «Обрезано» — ответы, где обращение завершилось по лимиту "
        "токенов (finish_reason=length); «Память» — по /api/ps. Все варианты получили одинаковые "
        "фрагменты (один поиск на вопрос"
        + (f", в среднем {sum(search_times) / len(search_times):.1f} с" if search_times else "")
        + ")."
    )
    if judged:
        judge = settings.get("judge") or {}
        lines.append(rme.judge_data_line(judge))
    else:
        lines.append("Качество не оценивалось (запуск без судьи): колонок качества нет.")
    if report.get("path"):
        lines.append(f"Отчёт: {report['path']}")
    return "\n".join(lines)


def format_question_detail(report: dict, number: int) -> str | None:
    """Ответы всех вариантов и повторов по вопросу рядом; None — такого номера в отчёте нет."""
    return rme.format_question_detail(report, number)


def question_numbers(report: dict) -> list[int]:
    return rme.question_numbers(report)
