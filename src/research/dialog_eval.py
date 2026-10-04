"""Чистые правила прогона длинных диалогов `/smart_agent` с рабочей задачей и справочными
материалами (скрипт scripts/dialog_eval.py, спека smart-agent-dialog-eval, design.md изменения
add-stage-aware-rag-and-dialog-runner, решение 4).

Отделён от скрипта по принципу проекта (как research/rag_compare_eval.py): здесь ПРАВИЛА —
формат сценария, проверки хода по двум снимкам задачи и признакам ответа, текст итога; а
обращения к SmartAgent и моделям, временная память и диск — в скрипте. Модуль не импортирует ни
openai, ни faiss, ни config, ни Telegram, поэтому все ветвления проверяются тестами
(tests/test_dialog_eval.py) без сети.

Снимок задачи — словарь `SmartAgent.get_working()` (или None, если задачи нет). Результат
хода — `TurnObservation`; проверки возвращают нарушения (обязательные) и наблюдения (для
оператора, на код выхода не влияют).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from agents import task_state

MIN_TURNS = 10
MAX_TURNS = 15

# Причины, по которым ключ данных или этап законно меняются «назад»: откат, отказ гейта,
# нарушение инвариантов, ручная команда (task_state.record_transition).
_LEGIT_BACKWARD_REASONS = (
    task_state.BY_ROLLBACK,
    task_state.BY_GATE_REJECTED,
    task_state.BY_INVARIANTS,
    task_state.BY_MANUAL,
)
_STAGE_ORDER = (
    task_state.STAGE_PLANNING,
    task_state.STAGE_EXECUTION,
    task_state.STAGE_VALIDATION,
    task_state.STAGE_DONE,
)

# Проверки (ключи нарушений и наблюдений в отчёте).
CHECK_EXPLANATION = "sources_or_note"
CHECK_CITATIONS = "citations_have_sources"
CHECK_GOAL = "goal_kept"
CHECK_KEYS = "keys_kept"
CHECK_ROLLBACK = "rollback_has_reason"
CHECK_ABSTAIN = "no_abstain_in_active_task"

# Какое объяснение использования материалов получил ход.
EXPL_SOURCES = "sources"
EXPL_UNVERIFIED = "unverified_citations"
EXPL_NO_MATERIALS = "no_materials_note"
EXPL_ABSTAIN = "abstain"
EXPL_FAILURE = "search_failure"
EXPL_STAGE_SKIP = "stage_skip"


class ScenarioError(ValueError):
    """Файл сценария не соответствует формату (сообщение — по-русски, для оператора)."""


@dataclass(frozen=True)
class Scenario:
    name: str
    task_type: str
    goal: str
    turns: tuple[str, ...]


def parse_scenario(raw: object, source: str = "сценарий") -> Scenario:
    """Проверяет разобранный JSON сценария: `{"name", "task": {"type", "goal"}, "turns": [...]}`.
    Число реплик — от MIN_TURNS до MAX_TURNS, тип задачи — из реестра автомата, цель и реплики —
    непустые строки. Любое нарушение — ScenarioError ДО обращений к модели."""
    if not isinstance(raw, dict):
        raise ScenarioError(f"{source}: ожидался JSON-объект.")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"{source}: нет непустого поля name.")
    task = raw.get("task")
    if not isinstance(task, dict):
        raise ScenarioError(f"{source}: нет объекта task с полями type и goal.")
    task_type, goal = task.get("type"), task.get("goal")
    if task_type not in task_state.SCENARIOS:
        known = ", ".join(task_state.SCENARIOS)
        raise ScenarioError(f"{source}: task.type должен быть одним из: {known}.")
    if not isinstance(goal, str) or not goal.strip():
        raise ScenarioError(f"{source}: нет непустой цели task.goal.")
    turns = raw.get("turns")
    if not isinstance(turns, list) or not all(isinstance(t, str) and t.strip() for t in turns):
        raise ScenarioError(f"{source}: turns — список непустых строк.")
    if not MIN_TURNS <= len(turns) <= MAX_TURNS:
        raise ScenarioError(
            f"{source}: реплик {len(turns)}, нужно от {MIN_TURNS} до {MAX_TURNS}."
        )
    return Scenario(
        name=name.strip(),
        task_type=task_type,
        goal=goal.strip(),
        turns=tuple(t.strip() for t in turns),
    )


def load_scenario(path: str) -> Scenario:
    try:
        with open(path, encoding="utf-8") as file:
            raw = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ScenarioError(f"{path}: не удалось прочитать JSON ({error}).") from error
    return parse_scenario(raw, source=path)


@dataclass
class TurnObservation:
    """Всё, что известно о ходе: снимки задачи до вопроса и после технического вызова автомата
    и признаки ответа (их собирает скрипт из SmartAgentAnswer)."""

    number: int
    question: str
    answer: str
    task_before: dict | None
    task_after: dict | None
    rag_active: bool  # слой материалов включён и индекс построен (до вопроса)
    sources: int = 0  # сколько фрагментов показано в источниках
    citations: int = 0  # сколько проверенных цитат показано
    citations_unverified: bool = False
    no_materials_note: bool = False
    stage_skip_note: bool = False
    abstained: bool = False
    search_failed: bool = False  # в warnings — предупреждение о сбое поиска
    error: str | None = None  # ход не состоялся: исключение при обращении к модели


@dataclass(frozen=True)
class Finding:
    check: str
    message: str


@dataclass
class TurnResult:
    observation: TurnObservation
    explanation: str | None
    violations: list[Finding] = field(default_factory=list)
    notes: list[Finding] = field(default_factory=list)


def explanation_of(obs: TurnObservation) -> str | None:
    """Чем ход объясняет использование материалов (спека smart-agent-rag, «На каждом ходу
    источники или явная строка»); None — ничем."""
    if obs.stage_skip_note:
        return EXPL_STAGE_SKIP
    if obs.abstained:
        return EXPL_ABSTAIN
    if obs.search_failed:
        return EXPL_FAILURE
    if obs.sources:
        return EXPL_UNVERIFIED if obs.citations_unverified and not obs.citations else EXPL_SOURCES
    if obs.no_materials_note:
        return EXPL_NO_MATERIALS
    return None


def _new_transitions(before: dict | None, after: dict | None) -> list[dict]:
    """Записи журнала переходов, появившиеся на ходу. Журнал ограничен по длине и записи
    могут совпадать (время — до секунды), поэтому разность считается как мультимножество."""
    remaining = list(task_state.transitions(before)) if before else []
    new: list[dict] = []
    for entry in task_state.transitions(after) if after else []:
        if entry in remaining:
            remaining.remove(entry)
        else:
            new.append(entry)
    return new


def _stage_index(task: dict) -> int:
    stage = task.get("stage")
    return _STAGE_ORDER.index(stage) if stage in _STAGE_ORDER else -1


def _filled_keys(task: dict) -> set[str]:
    return {key for key, value in (task.get("data") or {}).items() if str(value).strip()}


def check_turn(obs: TurnObservation, initial_goal: str) -> TurnResult:
    """Проверки одного хода (спека smart-agent-dialog-eval, «Проверки по каждому ходу»).
    Ход с ошибкой обращения к модели проверкам не подлежит — это ошибка прогона, а не бота."""
    result = TurnResult(observation=obs, explanation=explanation_of(obs))
    if obs.error:
        return result
    violations = result.violations

    # 1. Источники либо явная строка — пока слой включён и индекс есть.
    if obs.rag_active and result.explanation is None:
        violations.append(
            Finding(
                CHECK_EXPLANATION, "нет ни источников, ни пометки, ни отказа, ни предупреждения"
            )
        )
    if obs.citations and not obs.sources:
        violations.append(Finding(CHECK_CITATIONS, "цитаты показаны без списка источников"))

    # 2. Отказ «не знаю» без вызова модели на ходу активной задачи — нарушение на любом этапе
    # (решение 6 design.md): реплика вроде «предложи, что поменять» — не вопрос к корпусу.
    task_before = obs.task_before
    if (
        obs.abstained
        and task_before
        and not task_before.get("paused")
        and not task_state.is_done(task_before)
    ):
        violations.append(
            Finding(CHECK_ABSTAIN, "отказ «не знаю» без вызова модели на ходу активной задачи")
        )

    before, after = obs.task_before, obs.task_after
    if before is None or task_state.is_done(before):
        return result  # задачи нет или она завершена — цель и ключи не отслеживаются

    # 3. Цель задачи.
    if after is None:
        violations.append(Finding(CHECK_GOAL, "задача пропала"))
        return result
    if (after.get("goal") or "").strip() != initial_goal.strip():
        violations.append(Finding(CHECK_GOAL, f"цель изменилась: «{after.get('goal')}»"))

    new = _new_transitions(before, after)
    legit_backward = any(entry.get("by") in _LEGIT_BACKWARD_REASONS for entry in new)

    # 4. Ранее собранные ключи.
    lost = sorted(_filled_keys(before) - _filled_keys(after))
    if lost and not task_state.is_done(after) and not legit_backward:
        violations.append(
            Finding(CHECK_KEYS, "пропали ключи без отката: " + ", ".join(lost))
        )

    # 5. Откат этапа без записи причины.
    if 0 <= _stage_index(after) < _stage_index(before) and not legit_backward:
        violations.append(
            Finding(
                CHECK_ROLLBACK,
                f"этап вернулся {before.get('stage')} → {after.get('stage')} без записи причины",
            )
        )
    return result


# --------------------------------------------------------------------------- #
# Итог прогона
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioReport:
    scenario: Scenario
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None  # прогон прерван (ошибка обращения к модели)

    @property
    def violations(self) -> list[tuple[int, Finding]]:
        return [(t.observation.number, f) for t in self.turns for f in t.violations]

    @property
    def notes(self) -> list[tuple[int, Finding]]:
        return [(t.observation.number, f) for t in self.turns for f in t.notes]

    @property
    def failed(self) -> bool:
        return bool(self.error) or bool(self.violations)

    def to_dict(self) -> dict:
        return {
            "scenario": asdict(self.scenario),
            "error": self.error,
            "turns": [
                {
                    "observation": asdict(t.observation),
                    "explanation": t.explanation,
                    "violations": [asdict(f) for f in t.violations],
                    "notes": [asdict(f) for f in t.notes],
                }
                for t in self.turns
            ],
        }


def _stage_of(task: dict | None) -> str:
    return str(task.get("stage")) if task else "—"


def format_turn_line(result: TurnResult) -> str:
    obs = result.observation
    if obs.error:
        return f"{obs.number:>2}. ОШИБКА: {obs.error}"
    status = "нарушение" if result.violations else "ок"
    explanation = result.explanation or "нет"
    line = (
        f"{obs.number:>2}. {_stage_of(obs.task_before)} → {_stage_of(obs.task_after)}; "
        f"объяснение: {explanation}; {status}"
    )
    for finding in result.violations:
        line += f"\n      ✗ {finding.check}: {finding.message}"
    for finding in result.notes:
        line += f"\n      · {finding.check}: {finding.message}"
    return line


def format_summary(report: ScenarioReport) -> str:
    total = len(report.turns)
    verdict = "ПРОВАЛЕН" if report.failed else "пройден"
    lines = [
        f"Сценарий «{report.scenario.name}»: {verdict}; ходов {total} "
        f"из {len(report.scenario.turns)}, нарушений {len(report.violations)}, "
        f"наблюдений {len(report.notes)}."
    ]
    if report.error:
        lines.append(f"Прогон прерван: {report.error}")
    return "\n".join(lines)
