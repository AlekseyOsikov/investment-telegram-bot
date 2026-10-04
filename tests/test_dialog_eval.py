"""Тесты чистых правил прогона длинных диалогов (research/dialog_eval.py).

Только чистые функции — ни сети, ни LLM, ни SmartAgent. Прогон на живых моделях и Ollama
проверяется вручную (scripts/dialog_eval.py). Данные вымышленные.
"""

import pytest

from agents import task_state as ts
from research import dialog_eval as ev


def _raw(**overrides) -> dict:
    raw = {
        "name": "Пробный сценарий",
        "task": {"type": "portfolio", "goal": "накопить на дом"},
        "turns": [f"Реплика {i}" for i in range(1, 11)],
    }
    raw.update(overrides)
    return raw


def _task(stage=ts.STAGE_PLANNING, data=None, goal="накопить на дом", transitions=None, **extra):
    task = {"task_type": "portfolio", "goal": goal, "stage": stage, "data": dict(data or {})}
    if transitions:
        task["transitions"] = transitions
    task.update(extra)
    return task


def _transition(by, source="execution", target="planning", at="2026-01-01T10:00:00"):
    return {"at": at, "from": source, "to": target, "by": by, "note": ""}


def _obs(before, after, **kwargs) -> ev.TurnObservation:
    base = dict(number=1, question="Вопрос?", answer="Ответ.", rag_active=True)
    base.update(kwargs)
    return ev.TurnObservation(task_before=before, task_after=after, **base)


def _checks(result: ev.TurnResult) -> list[str]:
    return [finding.check for finding in result.violations]


# --- формат сценария ---------------------------------------------------------------------------


def test_parse_scenario_ok():
    scenario = ev.parse_scenario(_raw())
    assert scenario.task_type == "portfolio" and len(scenario.turns) == 10


@pytest.mark.parametrize("count", [9, 16])
def test_parse_scenario_rejects_wrong_turn_count(count):
    with pytest.raises(ev.ScenarioError, match="от 10 до 15"):
        ev.parse_scenario(_raw(turns=["Реплика"] * count))


@pytest.mark.parametrize(
    "raw",
    [
        [],
        _raw(name=""),
        _raw(task={"type": "unknown", "goal": "цель"}),
        _raw(task={"type": "portfolio", "goal": " "}),
        _raw(task="portfolio"),
        _raw(turns=["ok"] * 9 + [""]),
        _raw(turns="не список"),
    ],
)
def test_parse_scenario_rejects_malformed(raw):
    with pytest.raises(ev.ScenarioError):
        ev.parse_scenario(raw)


# --- источники или явная строка ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"sources": 2, "citations": 1}, ev.EXPL_SOURCES),
        ({"sources": 2, "citations_unverified": True}, ev.EXPL_UNVERIFIED),
        ({"no_materials_note": True}, ev.EXPL_NO_MATERIALS),
        ({"abstained": True}, ev.EXPL_ABSTAIN),
        ({"search_failed": True}, ev.EXPL_FAILURE),
        ({"stage_skip_note": True}, ev.EXPL_STAGE_SKIP),
        ({}, None),
    ],
)
def test_explanation_of(flags, expected):
    assert ev.explanation_of(_obs(None, None, **flags)) == expected


def test_turn_without_explanation_is_a_violation_only_when_rag_active():
    assert _checks(ev.check_turn(_obs(None, None), "цель")) == [ev.CHECK_EXPLANATION]
    assert _checks(ev.check_turn(_obs(None, None, rag_active=False), "цель")) == []


def test_citations_without_sources_is_a_violation():
    result = ev.check_turn(_obs(None, None, citations=1, no_materials_note=True), "цель")
    assert ev.CHECK_CITATIONS in _checks(result)


# --- отказ «не знаю» ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", [ts.STAGE_PLANNING, ts.STAGE_EXECUTION, ts.STAGE_VALIDATION])
def test_abstain_on_active_task_is_a_violation_on_any_stage(stage):
    before = _task(stage=stage)
    result = ev.check_turn(_obs(before, before, abstained=True), "накопить на дом")
    assert _checks(result) == [ev.CHECK_ABSTAIN]


def test_abstain_on_paused_or_done_task_is_not_a_violation():
    paused = _task(stage=ts.STAGE_EXECUTION, paused=True)
    assert ev.check_turn(_obs(paused, paused, abstained=True), "накопить на дом").violations == []
    done = _task(stage=ts.STAGE_DONE)
    assert ev.check_turn(_obs(done, done, abstained=True), "накопить на дом").violations == []


def test_abstain_without_task_is_not_a_violation():
    result = ev.check_turn(_obs(None, None, abstained=True), "цель")
    assert result.violations == [] and result.notes == []


# --- цель, ключи, откаты -----------------------------------------------------------------------


def test_goal_kept_and_changed():
    before = _task(data={"horizon": "10 лет"})
    ok = ev.check_turn(_obs(before, _task(data={"horizon": "10 лет"}), no_materials_note=True),
                       "накопить на дом")
    assert ok.violations == []

    changed = ev.check_turn(
        _obs(before, _task(goal="другая цель", data={"horizon": "10 лет"}), no_materials_note=True),
        "накопить на дом",
    )
    assert _checks(changed) == [ev.CHECK_GOAL]


def test_task_lost_is_a_goal_violation():
    result = ev.check_turn(_obs(_task(), None, no_materials_note=True), "накопить на дом")
    assert _checks(result) == [ev.CHECK_GOAL]


def test_lost_key_without_rollback_is_a_violation():
    before = _task(data={"horizon": "10 лет", "risk": "умеренный"})
    after = _task(data={"horizon": "10 лет"})
    result = ev.check_turn(_obs(before, after, no_materials_note=True), "накопить на дом")
    assert _checks(result) == [ev.CHECK_KEYS]
    assert "risk" in result.violations[0].message


def test_lost_key_with_recorded_rollback_is_fine():
    before = _task(stage=ts.STAGE_EXECUTION, data={"allocation": "акции 60%"})
    after = _task(
        stage=ts.STAGE_PLANNING,
        data={},
        transitions=[_transition(ts.BY_ROLLBACK)],
    )
    result = ev.check_turn(_obs(before, after, no_materials_note=True), "накопить на дом")
    assert result.violations == []


def test_rollback_log_entry_already_in_before_does_not_count_as_new():
    old = _transition(ts.BY_ROLLBACK)
    before = _task(data={"risk": "умеренный"}, transitions=[old])
    after = _task(data={}, transitions=[old])
    result = ev.check_turn(_obs(before, after, no_materials_note=True), "накопить на дом")
    assert _checks(result) == [ev.CHECK_KEYS]


def test_stage_regression_without_reason_is_a_violation():
    before = _task(stage=ts.STAGE_VALIDATION)
    after = _task(stage=ts.STAGE_EXECUTION)
    result = ev.check_turn(_obs(before, after, no_materials_note=True), "накопить на дом")
    assert _checks(result) == [ev.CHECK_ROLLBACK]


def test_stage_regression_with_invariants_reason_is_fine():
    before = _task(stage=ts.STAGE_VALIDATION)
    after = _task(stage=ts.STAGE_EXECUTION, transitions=[_transition(ts.BY_INVARIANTS)])
    result = ev.check_turn(_obs(before, after, no_materials_note=True), "накопить на дом")
    assert result.violations == []


def test_forward_move_and_done_task_are_not_checked():
    forward = ev.check_turn(
        _obs(_task(), _task(stage=ts.STAGE_EXECUTION), no_materials_note=True), "накопить на дом"
    )
    assert forward.violations == []

    done = _task(stage=ts.STAGE_DONE, data={"allocation": "x"})
    finished = ev.check_turn(_obs(done, _task(goal="новая", stage=ts.STAGE_PLANNING),
                                  no_materials_note=True), "накопить на дом")
    assert finished.violations == []


def test_turn_with_model_error_is_not_checked():
    result = ev.check_turn(_obs(_task(), None, error="таймаут"), "накопить на дом")
    assert result.violations == [] and result.explanation is None


# --- итог --------------------------------------------------------------------------------------


def test_report_failed_flags_and_summary():
    scenario = ev.parse_scenario(_raw())
    report = ev.ScenarioReport(scenario=scenario)
    ok = ev.check_turn(_obs(None, None, no_materials_note=True), "цель")
    report.turns.append(ok)
    assert not report.failed
    assert "пройден" in ev.format_summary(report)

    bad = ev.check_turn(_obs(None, None, number=2), "цель")
    report.turns.append(bad)
    assert report.failed and report.violations[0][0] == 2
    assert "ПРОВАЛЕН" in ev.format_summary(report)
    assert "✗" in ev.format_turn_line(bad)

    data = report.to_dict()
    assert data["turns"][1]["violations"][0]["check"] == ev.CHECK_EXPLANATION


def test_report_with_error_is_failed():
    report = ev.ScenarioReport(scenario=ev.parse_scenario(_raw()), error="ошибка API")
    assert report.failed and "прерван" in ev.format_summary(report)
