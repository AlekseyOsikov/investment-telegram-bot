"""Тесты конечного автомата рабочей задачи (agents/task_state.py).

Модуль автомата не знает ни про Telegram, ни про openai — только про словарь
задачи, поэтому здесь нет ни сети, ни моков LLM: проверяются ровно те правила
переходов, на которые опирается /smart_agent. Технический вызов LLM в этих тестах
представлен его РАЗОБРАННЫМ ответом (dict), как его и получает
apply_state_response.
"""

from agents import task_state as ts

PORTFOLIO_INPUTS = {
    "goal_type": "накопить на пенсию",
    "horizon": "15 лет",
    "risk": "умеренный",
    "constraints": "без криптовалют",
}


# --------------------------------------------------------------------------- #
# Вспомогательные шаги: состояние собирается через публичный API автомата, а не
# руками, чтобы тесты проверяли реальный путь задачи.
# --------------------------------------------------------------------------- #


def apply(task: dict, data: dict | None = None, stage: str | None = None) -> ts.StateChange:
    parsed: dict = {}
    if data is not None:
        parsed["data_updates"] = data
    if stage is not None:
        parsed["stage"] = stage
    return ts.apply_state_response(task, parsed)


def portfolio_at_execution() -> dict:
    """Портфельная задача, прошедшая профилирование штатным путём: параметры →
    сводка вводных → подтверждение."""
    task = ts.new_task("portfolio", "портфель на пенсию")
    task = apply(task, PORTFOLIO_INPUTS).task
    task = apply(task, {ts.BRIEF_KEY: "пенсия, 15 лет, умеренный риск, без крипты"}).task
    change = apply(task, {ts.BRIEF_VERDICT_KEY: ts.BRIEF_CONFIRMED})
    assert change.task["stage"] == ts.STAGE_EXECUTION
    return change.task


def portfolio_at_validation() -> dict:
    change = apply(portfolio_at_execution(), {"allocation": "акции 40%, облигации 60%"})
    assert change.task["stage"] == ts.STAGE_VALIDATION
    return change.task


# --------------------------------------------------------------------------- #
# Нельзя начать работу до подтверждения вводных
# --------------------------------------------------------------------------- #


def test_planning_is_the_initial_stage():
    task = ts.new_task("portfolio", "портфель")
    assert task["stage"] == ts.STAGE_PLANNING
    assert ts.missing_keys(task) == [
        "goal_type",
        "horizon",
        "risk",
        "constraints",
        ts.BRIEF_KEY,
        ts.BRIEF_VERDICT_KEY,
    ]


def test_brief_and_verdict_rejected_before_inputs_collected():
    """Сводку вводных нельзя проговорить, пока вводные не собраны, а подтвердить —
    пока сводки нет: иначе гейт закрывался бы первым же ходом."""
    task = ts.new_task("portfolio", "портфель")
    change = apply(
        task,
        {
            "goal_type": "пенсия",
            ts.BRIEF_KEY: "всё понятно",
            ts.BRIEF_VERDICT_KEY: ts.BRIEF_CONFIRMED,
        },
    )

    assert change.task["data"] == {"goal_type": "пенсия"}
    assert change.task["stage"] == ts.STAGE_PLANNING
    assert len(change.rejected) == 2
    assert all("рано" in reason for reason in change.rejected)


def test_execution_unreachable_until_brief_confirmed():
    """Ключевое требование: работа не начинается, пока пользователь не подтвердил
    сводку вводных."""
    task = apply(ts.new_task("portfolio", "портфель"), PORTFOLIO_INPUTS).task
    assert task["stage"] == ts.STAGE_PLANNING
    assert ts.missing_keys(task) == [ts.BRIEF_KEY, ts.BRIEF_VERDICT_KEY]

    task = apply(task, {ts.BRIEF_KEY: "пенсия, 15 лет, умеренный риск"}).task
    assert task["stage"] == ts.STAGE_PLANNING
    assert ts.missing_keys(task) == [ts.BRIEF_VERDICT_KEY]

    change = apply(task, {ts.BRIEF_VERDICT_KEY: ts.BRIEF_CONFIRMED})
    assert change.task["stage"] == ts.STAGE_EXECUTION
    assert change.transition == "планирование → формирование портфеля"


def test_allocation_rejected_while_still_planning():
    """Попытка выдать результат до подтверждения вводных: ключ чужого этапа
    отбрасывается, задача остаётся на профилировании."""
    task = apply(ts.new_task("portfolio", "портфель"), PORTFOLIO_INPUTS).task
    change = apply(task, {"allocation": "акции 100%"}, stage=ts.STAGE_EXECUTION)

    assert change.task["stage"] == ts.STAGE_PLANNING
    assert "allocation" not in change.task["data"]
    assert any("не пункт этапа" in reason for reason in change.rejected)
    assert any("не хватает" in reason for reason in change.rejected)


def test_brief_corrected_resets_the_gate():
    task = apply(ts.new_task("portfolio", "портфель"), PORTFOLIO_INPUTS).task
    task = apply(task, {ts.BRIEF_KEY: "пенсия, 15 лет"}).task

    change = apply(task, {ts.BRIEF_VERDICT_KEY: ts.BRIEF_CORRECTED})

    assert change.gate_reset is True
    assert change.task["stage"] == ts.STAGE_PLANNING
    assert ts.BRIEF_KEY not in change.task["data"]
    assert ts.BRIEF_VERDICT_KEY not in change.task["data"]
    assert change.task["awaiting_correction"] is True
    assert change.task["transitions"][-1]["by"] == ts.BY_GATE_REJECTED


def test_asset_scenario_has_no_input_gate():
    """Справочный разбор идёт к работе сразу по собранным параметрам — гейт есть
    только там, где результат является рекомендацией по структуре."""
    assert ts.SCENARIOS["asset"].stage(ts.STAGE_PLANNING).gate is None

    task = ts.new_task("asset", "разбор ОФЗ")
    change = apply(
        task,
        {"asset": "ОФЗ 26238", "horizon": "3 года", "role_in_portfolio": "защитная часть"},
    )
    assert change.task["stage"] == ts.STAGE_EXECUTION


# --------------------------------------------------------------------------- #
# Нельзя завершить задачу без приёмки результата
# --------------------------------------------------------------------------- #


def test_verdict_rejected_outside_validation():
    """Раньше модель могла вернуть результат и приёмку одним ответом, и задача
    уходила в done, минуя проверку (и ревизора инвариантов)."""
    change = apply(
        portfolio_at_execution(),
        {"allocation": "акции 40%, облигации 60%", ts.VERDICT_KEY: ts.VERDICT_ACCEPTED},
    )

    assert change.task["stage"] == ts.STAGE_VALIDATION
    assert ts.VERDICT_KEY not in change.task["data"]
    assert any("не пункт этапа" in reason for reason in change.rejected)


def test_done_requires_explicit_acceptance():
    task = portfolio_at_validation()
    assert ts.missing_keys(task) == [ts.VERDICT_KEY]

    change = apply(task, {ts.VERDICT_KEY: ts.VERDICT_ACCEPTED})
    assert change.task["stage"] == ts.STAGE_DONE
    assert change.transition == "проверка → завершено"


def test_unknown_verdict_value_keeps_task_on_validation():
    change = apply(portfolio_at_validation(), {ts.VERDICT_KEY: "вроде согласен"})

    assert change.task["stage"] == ts.STAGE_VALIDATION
    assert ts.VERDICT_KEY not in change.task["data"]
    assert any("не из списка" in reason for reason in change.rejected)


def test_changes_requested_rolls_back_and_keeps_result():
    change = apply(portfolio_at_validation(), {ts.VERDICT_KEY: ts.VERDICT_CHANGES_REQUESTED})
    task = change.task

    assert task["stage"] == ts.STAGE_EXECUTION
    # Результат сохраняется: агент правит прежний вариант, а не сочиняет заново.
    assert task["data"]["allocation"] == "акции 40%, облигации 60%"
    assert ts.VERDICT_KEY not in task["data"]
    assert task["awaiting_correction"] is True
    assert task["transitions"][-1]["by"] == ts.BY_ROLLBACK

    # Флаг ожидания правки блокирует движение вперёд, пока правка не придёт.
    assert apply(task, {}).task["stage"] == ts.STAGE_EXECUTION
    assert apply(task, {"allocation": "акции 30%, облигации 70%"}).task["stage"] == (
        ts.STAGE_VALIDATION
    )


def test_rollback_to_planning_requires_new_confirmation():
    """Вернулись править вводные — прежнее подтверждение сводки больше не значит
    ничего, иначе задача уехала бы вперёд по устаревшему согласию."""
    change = apply(portfolio_at_execution(), {}, stage=ts.STAGE_PLANNING)
    task = change.task

    assert task["stage"] == ts.STAGE_PLANNING
    assert ts.BRIEF_VERDICT_KEY not in task["data"]
    assert ts.BRIEF_KEY not in task["data"]
    assert task["data"]["horizon"] == "15 лет"

    task = apply(task, {"horizon": "20 лет"}).task
    assert task["stage"] == ts.STAGE_PLANNING
    assert ts.missing_keys(task) == [ts.BRIEF_KEY, ts.BRIEF_VERDICT_KEY]


# --------------------------------------------------------------------------- #
# Попытки перейти в недопустимое состояние
# --------------------------------------------------------------------------- #


def test_forward_jump_over_stage_rejected():
    task = ts.new_task("portfolio", "портфель")
    change = apply(task, {}, stage=ts.STAGE_DONE)

    assert change.task["stage"] == ts.STAGE_PLANNING
    assert any("не разрешён" in reason for reason in change.rejected)
    assert change.task["transitions"][-1]["by"] == ts.BY_REJECTED


def test_nonexistent_stage_rejected():
    change = apply(ts.new_task("portfolio", "портфель"), {}, stage="deploy")

    assert change.task["stage"] == ts.STAGE_PLANNING
    assert any("не существует" in reason for reason in change.rejected)


def test_manual_transition_respects_graph():
    task = portfolio_at_execution()

    assert ts.manual_transition(task, ts.STAGE_DONE) is False
    assert task["stage"] == ts.STAGE_EXECUTION
    assert task["transitions"][-1]["by"] == ts.BY_REJECTED

    assert ts.manual_transition(task, ts.STAGE_VALIDATION) is True
    assert task["stage"] == ts.STAGE_VALIDATION
    assert task["transitions"][-1]["by"] == ts.BY_MANUAL


def test_manual_transition_forward_ignores_missing_keys():
    """Предохранитель: застрявшую задачу можно сдвинуть вручную, даже если обычные
    обязательные пункты не собраны (командный слой предупреждает об этом отдельно).
    Сценарий без гейта — asset."""
    task = ts.new_task("asset", "разбор ОФЗ")
    assert ts.is_forward_stage(task, ts.STAGE_EXECUTION) is True
    assert ts.missing_keys(task)

    assert ts.manual_transition_block(task, ts.STAGE_EXECUTION) is None
    assert ts.manual_transition(task, ts.STAGE_EXECUTION) is True
    assert task["stage"] == ts.STAGE_EXECUTION


def test_manual_transition_cannot_jump_over_a_closed_gate():
    """Предохранитель сдвигает задачу, но не решает за человека: гейт — это решение
    пользователя, и ручной перевод его не заменяет."""
    task = apply(ts.new_task("portfolio", "портфель"), PORTFOLIO_INPUTS).task
    task = apply(task, {ts.BRIEF_KEY: "пенсия, 15 лет"}).task

    block = ts.manual_transition_block(task, ts.STAGE_EXECUTION)
    assert block is not None
    assert ts.BRIEF_VERDICT_KEY in block  # в подсказке назван и ключ, и команда

    assert ts.manual_transition(task, ts.STAGE_EXECUTION) is False
    assert task["stage"] == ts.STAGE_PLANNING
    assert task["transitions"][-1]["by"] == ts.BY_REJECTED

    # Явное решение человека (ответ в диалоге или /smart_agent_task_set) гейт
    # закрывает — и тогда ручной перевод снова возможен.
    task["data"][ts.BRIEF_VERDICT_KEY] = ts.BRIEF_CONFIRMED
    assert ts.manual_transition_block(task, ts.STAGE_EXECUTION) is None
    assert ts.manual_transition(task, ts.STAGE_EXECUTION) is True


def test_manual_transition_back_is_never_blocked_by_a_gate():
    """Назад гейт не мешает: вернуться и переделать можно всегда."""
    task = portfolio_at_execution()

    assert ts.manual_transition_block(task, ts.STAGE_PLANNING) is None
    assert ts.manual_transition(task, ts.STAGE_PLANNING) is True
    assert task["stage"] == ts.STAGE_PLANNING
    # Откат снял гейт целевого этапа — подтверждать вводные придётся заново.
    assert ts.BRIEF_VERDICT_KEY not in task["data"]


def test_transitions_history_is_capped():
    task = portfolio_at_validation()
    for index in range(ts.TRANSITIONS_MAX + 5):
        ts.record_transition(
            task, ts.STAGE_VALIDATION, ts.STAGE_VALIDATION, ts.BY_REJECTED, str(index)
        )

    assert len(task["transitions"]) == ts.TRANSITIONS_MAX
    assert task["transitions"][-1]["note"] == str(ts.TRANSITIONS_MAX + 4)


# --------------------------------------------------------------------------- #
# Инварианты: проверка результата привязана к результату, а не к переходу
# --------------------------------------------------------------------------- #


def test_invariant_check_needed_after_manual_switch_to_validation():
    task = portfolio_at_execution()
    task["data"]["allocation"] = "акции 40%, облигации 60%"
    assert ts.needs_invariant_check(task) is False

    ts.manual_transition(task, ts.STAGE_VALIDATION)
    assert ts.needs_invariant_check(task) is True

    ts.mark_invariants_checked(task)
    assert ts.needs_invariant_check(task) is False

    # Новый результат — новая проверка.
    task["data"]["allocation"] = "акции 80%, облигации 20%"
    assert ts.needs_invariant_check(task) is True


def test_invariant_violations_roll_task_back():
    task = portfolio_at_validation()
    violations = [{"index": 1, "text": "без криптовалют", "why": "в структуре есть биткоин"}]

    updated, transition = ts.apply_invariant_violations(task, violations)

    assert updated["stage"] == ts.STAGE_EXECUTION
    assert updated["data"]["allocation"] == "акции 40%, облигации 60%"
    assert updated["awaiting_correction"] is True
    assert ts.invariant_violations(updated) == violations
    assert transition == "проверка → формирование портфеля"
    assert updated["transitions"][-1]["by"] == ts.BY_INVARIANTS

    # Новый результат снимает претензии и снова открывает путь вперёд.
    change = apply(updated, {"allocation": "акции 40%, облигации 60%, без крипты"})
    assert change.task["stage"] == ts.STAGE_VALIDATION
    assert ts.invariant_violations(change.task) == []


# --------------------------------------------------------------------------- #
# Пауза и продолжение
# --------------------------------------------------------------------------- #


def test_context_names_the_next_stage():
    """Модель должна знать, что будет дальше: без этого она сочиняет пользователю
    собственные этапы («дальше — наполнение блоков»), которых в автомате нет."""
    execution = ts.build_context_message(portfolio_at_execution())
    assert "Следующий этап: проверка." in execution
    assert "Названий этапов не выдумывай" in execution

    validation = ts.build_context_message(portfolio_at_validation())
    assert "Следующий этап: завершено." in validation

    done = apply(portfolio_at_validation(), {ts.VERDICT_KEY: ts.VERDICT_ACCEPTED}).task
    assert "Это последний этап." in ts.build_context_message(done)


def test_pause_is_orthogonal_to_stage():
    task = portfolio_at_execution()
    task["paused"] = True
    task["paused_at"] = "2026-09-20T12:00:00"
    task["expected_action"] = "предложить структуру портфеля"

    assert task["stage"] == ts.STAGE_EXECUTION
    assert ts.state_line(task).startswith("⏸")
    assert "ЗАДАЧА НА ПАУЗЕ" in ts.build_context_message(task)
    assert "не продолжай её" in ts.build_context_message(task)


def test_resume_summary_is_built_from_state_only():
    """Сводка «где остановились» собирается из сохранённого состояния — именно это и
    позволяет продолжить задачу после паузы, ничего не переспрашивая."""
    task = portfolio_at_validation()
    task["paused"] = True
    task["paused_at"] = "2026-09-20T12:00:00"
    task["step"] = "ждём решения пользователя"

    summary = ts.describe_state(task)

    assert "Составление портфеля" in summary
    assert "проверка" in summary
    assert "на паузе с 2026-09-20T12:00:00" in summary
    assert "акции 40%, облигации 60%" in summary
    assert "умеренный" in summary
    assert "История переходов:" in summary


# --------------------------------------------------------------------------- #
# Загрузка задач из файла памяти
# --------------------------------------------------------------------------- #


def test_sanitize_migrates_pre_state_machine_task():
    task = ts.sanitize_task({"goal": "портфель", "status": "active", "data": {"horizon": "5 лет"}})

    assert task["task_type"] == ts.DEFAULT_SCENARIO
    assert task["stage"] == ts.STAGE_PLANNING
    assert task["data"]["horizon"] == "5 лет"
    assert task["transitions"] == []
    # Задача осталась на профилировании — сводку она доберёт штатным путём.
    assert ts.BRIEF_VERDICT_KEY not in task["data"]


def test_sanitize_marks_gate_passed_for_tasks_beyond_planning():
    """Задачам, которые уже работают, сводку задним числом не показать — гейт
    считается пройденным, иначе откат назад потребовал бы подтвердить то, чего
    пользователь никогда не видел."""
    task = ts.sanitize_task(
        {
            "task_type": "portfolio",
            "goal": "портфель",
            "stage": ts.STAGE_VALIDATION,
            "data": {**PORTFOLIO_INPUTS, "allocation": "акции 50%, облигации 50%"},
        }
    )

    assert task["data"][ts.BRIEF_VERDICT_KEY] == ts.BRIEF_CONFIRMED
    assert task["data"][ts.BRIEF_KEY]
    assert ts.missing_keys(task) == [ts.VERDICT_KEY]


def test_sanitize_falls_back_on_unknown_stage():
    task = ts.sanitize_task({"task_type": "asset", "goal": "разбор", "stage": "review-2"})

    assert task["task_type"] == "asset"
    assert task["stage"] == ts.STAGE_PLANNING


# --------------------------------------------------------------------------- #
# Признак «поиск по материалам разрешён» (add-stage-aware-rag-and-dialog-runner)
# --------------------------------------------------------------------------- #


def test_rag_search_flag_by_scenario_and_stage():
    expected = {
        ("portfolio", ts.STAGE_PLANNING): False,
        ("portfolio", ts.STAGE_EXECUTION): True,
        ("portfolio", ts.STAGE_VALIDATION): False,
        ("review", ts.STAGE_PLANNING): False,
        ("review", ts.STAGE_EXECUTION): True,
        ("review", ts.STAGE_VALIDATION): False,
        ("asset", ts.STAGE_PLANNING): True,
        ("asset", ts.STAGE_EXECUTION): True,
        ("asset", ts.STAGE_VALIDATION): True,
    }
    for (scenario, stage), allowed in expected.items():
        assert ts.SCENARIOS[scenario].stage(stage).rag_search is allowed, (scenario, stage)
    # Завершающий этап и любой новый этап по умолчанию — поиск разрешён.
    for scenario in ts.SCENARIOS.values():
        assert scenario.stage(ts.STAGE_DONE).rag_search is True


def test_search_allowed_follows_current_stage():
    task = ts.new_task("portfolio", "портфель на пенсию")
    assert ts.search_allowed(task) is False  # planning

    assert ts.search_allowed(portfolio_at_execution()) is True

    asset = ts.new_task("asset", "разбор облигаций")
    assert ts.search_allowed(asset) is True


def test_search_allowed_ignores_flag_without_active_task():
    assert ts.search_allowed(None) is True

    paused = ts.new_task("portfolio", "портфель на пенсию")
    paused["paused"] = True
    assert ts.search_allowed(paused) is True

    done = ts.new_task("portfolio", "портфель на пенсию")
    done["stage"] = ts.STAGE_DONE
    assert ts.search_allowed(done) is True


def test_search_allowed_survives_broken_state():
    broken = {"task_type": "unknown", "stage": "no-such-stage"}
    # Неизвестный сценарий откатывается на portfolio/planning — главное, без исключения.
    assert ts.search_allowed(broken) in (True, False)
    assert ts.search_allowed({"task_type": 123, "stage": None}) in (True, False)


def test_rag_search_flag_does_not_change_transitions():
    task = ts.new_task("portfolio", "портфель на пенсию")
    assert ts.missing_keys(task) == [
        "goal_type",
        "horizon",
        "risk",
        "constraints",
        ts.BRIEF_KEY,
        ts.BRIEF_VERDICT_KEY,
    ]
    task = apply(task, PORTFOLIO_INPUTS).task
    assert task["stage"] == ts.STAGE_PLANNING  # без сводки и подтверждения — не уходит


# --------------------------------------------------------------------------- #
# Дополнение: реестр, разбор файла памяти, граничные ветки и тексты автомата
# --------------------------------------------------------------------------- #

import datetime as _dt  # noqa: E402

import pytest  # noqa: E402


def _planning_stage(scenario: str = "portfolio") -> ts.TaskStage:
    return ts.SCENARIOS[scenario].stage(ts.STAGE_PLANNING)


def _violations() -> list[dict]:
    return [{"index": 2, "text": "без криптовалют", "why": "в структуре есть токены"}]


def test_keys_before_lists_required_keys_ahead_of_the_given_one():
    stage = _planning_stage()
    assert stage.keys_before("risk") == ("goal_type", "horizon")
    assert stage.keys_before("goal_type") == ()
    assert stage.keys_before("нет такого") == ()
    assert stage.keys_before("preferred_assets") == ()  # необязательный ключ


def test_scenario_stage_index_is_position_or_minus_one():
    scenario = ts.SCENARIOS["portfolio"]
    assert scenario.stage_index(ts.STAGE_PLANNING) == 0
    assert scenario.stage_index(ts.STAGE_DONE) == len(scenario.stages) - 1
    assert scenario.stage_index("нет такого") == -1
    assert scenario.stage("нет такого") is None


@pytest.mark.parametrize("raw", [None, "x", 5, {"a": 1}])
def test_sanitize_transitions_of_non_list_is_empty(raw):
    assert ts._sanitize_transitions(raw) == []


def test_sanitize_transitions_skips_junk_coerces_values_and_keeps_last_ones():
    entries = [
        {"at": "t", "from": "planning", "to": "execution", "by": "auto", "note": 7},
        "мусор",
        None,
        {"from": "a"},
    ]
    result = ts._sanitize_transitions(entries)
    assert result == [
        {"at": "t", "from": "planning", "to": "execution", "by": "auto", "note": "7"},
        {"at": "", "from": "a", "to": "", "by": "", "note": ""},
    ]
    many = [{"note": str(i)} for i in range(ts.TRANSITIONS_MAX + 5)]
    kept = ts._sanitize_transitions(many)
    assert len(kept) == ts.TRANSITIONS_MAX
    assert kept[-1]["note"] == str(ts.TRANSITIONS_MAX + 4)


@pytest.mark.parametrize("raw", [None, "x", {"index": 1}])
def test_sanitize_violations_of_non_list_is_empty(raw):
    assert ts._sanitize_violations(raw) == []


def test_sanitize_violations_skips_junk_and_bad_index_and_coerces_text():
    raw = [
        {"index": "2", "text": "запрет", "why": "причина"},
        {"index": 3, "text": None},
        {"index": "не число", "text": "x"},
        {"text": "без номера"},
        "строка",
    ]
    assert ts._sanitize_violations(raw) == [
        {"index": 2, "text": "запрет", "why": "причина"},
        {"index": 3, "text": "", "why": ""},
    ]


def test_as_text_normalises_every_kind_of_value():
    assert ts._as_text("строка") == "строка"
    assert ts._as_text(None) == ""
    assert ts._as_text(15) == "15"
    assert ts._as_text(True) == "True"
    assert ts._as_text({"акции": 40}) == '{"акции": 40}'
    assert ts._as_text(["а", "б"]) == '["а", "б"]'


def test_search_allowed_with_unhashable_scenario_is_true_not_an_exception():
    assert ts.search_allowed({"task_type": [], "stage": "planning"}) is True


def test_search_allowed_per_stage_of_scenarios():
    assert ts.search_allowed(ts.new_task("portfolio", "g")) is False
    assert ts.search_allowed(ts.new_task("asset", "g")) is True
    assert ts.search_allowed(portfolio_at_execution()) is True
    assert ts.search_allowed(portfolio_at_validation()) is False


def test_needs_invariant_check_only_for_unchecked_result_on_validation():
    assert ts.needs_invariant_check(portfolio_at_execution()) is False
    empty_validation = ts.new_task("portfolio", "g")
    empty_validation["stage"] = ts.STAGE_VALIDATION
    assert ts.needs_invariant_check(empty_validation) is False

    task = portfolio_at_validation()
    assert ts.needs_invariant_check(task) is True
    ts.mark_invariants_checked(task)
    assert ts.needs_invariant_check(task) is False
    task["data"]["allocation"] = "акции 30%, облигации 70%"
    assert ts.needs_invariant_check(task) is True


def test_result_payload_uses_labels_of_auto_stage_keys_only():
    task = portfolio_at_validation()
    assert ts.result_payload(task) == {
        ts.key_label("allocation"): "акции 40%, облигации 60%"
    }
    assert ts.key_label("нет такого ключа") == "нет такого ключа"


def test_manual_stage_options_are_graph_neighbours_with_labels():
    assert ts.manual_stage_options(portfolio_at_execution()) == [
        (ts.STAGE_VALIDATION, "проверка"),
        (ts.STAGE_PLANNING, "планирование"),
    ]
    assert ts.manual_stage_options(ts.new_task("asset", "g")) == [
        (ts.STAGE_EXECUTION, "разбор")
    ]


def test_short_summary_counts_filled_data_and_marks_pause():
    task = ts.new_task("portfolio", "пенсия")
    task["data"] = {"goal_type": "пенсия", "horizon": "", "risk": "умеренный"}
    line = ts.short_summary(task)
    assert "Составление портфеля" in line
    assert "«пенсия»" in line
    assert "этап: планирование" in line
    assert "данных: 2" in line
    assert "на паузе" not in line
    task["paused"] = True
    assert ts.short_summary(task).endswith("⏸ на паузе")


def test_short_summary_without_goal_shows_dash():
    assert "«—»" in ts.short_summary(ts.new_task("asset", ""))


def test_normalize_skips_empty_values_and_coerces_structures():
    stage = _planning_stage("asset")
    accepted, rejected = ts.normalize_data_updates(
        stage, {}, {"asset": "  ", "horizon": ["3 года"], "role_in_portfolio": None}
    )
    assert accepted == {"horizon": '["3 года"]'}
    assert rejected == []


def test_forward_target_is_none_when_gate_holds_a_refusal_value():
    task = portfolio_at_validation()
    task["data"][ts.VERDICT_KEY] = ts.VERDICT_CHANGES_REQUESTED
    scenario = ts.scenario_of(task)
    assert ts.missing_keys(task) == []
    assert ts._forward_target(scenario, task) is None
    task["data"][ts.VERDICT_KEY] = ts.VERDICT_ACCEPTED
    assert ts._forward_target(scenario, task) == ts.STAGE_DONE


def test_reset_gate_on_stage_without_gate_changes_nothing():
    task = ts.new_task("asset", "g")
    task["data"] = {"asset": "ОФЗ"}
    ts._reset_gate(_planning_stage("asset"), task)
    assert task["data"] == {"asset": "ОФЗ"}
    assert task["awaiting_correction"] is False
    assert task["transitions"] == []


def test_rejected_stage_request_reasons():
    scenario = ts.SCENARIOS["asset"]
    stage = _planning_stage("asset")
    task = ts.new_task("asset", "g")

    assert ts._rejected_stage_request(scenario, stage, task, None) == []
    assert ts._rejected_stage_request(scenario, stage, task, 5) == []
    assert ts._rejected_stage_request(scenario, stage, task, "  ") == []
    assert ts._rejected_stage_request(scenario, stage, task, ts.STAGE_PLANNING) == []

    unknown = ts._rejected_stage_request(scenario, stage, task, "полёт")
    assert "не существует" in unknown[0]

    not_allowed = ts._rejected_stage_request(scenario, stage, task, ts.STAGE_DONE)
    assert "не разрешён" in not_allowed[0]

    missing = ts._rejected_stage_request(scenario, stage, task, ts.STAGE_EXECUTION)
    assert "не хватает" in missing[0]
    assert ts.key_label("asset") in missing[0]

    task["data"] = {"asset": "ОФЗ", "horizon": "3 года", "role_in_portfolio": "защита"}
    forward = ts._rejected_stage_request(scenario, stage, task, ts.STAGE_EXECUTION)
    assert "автомат двигается сам" in forward[0]


def test_roll_back_to_unknown_stage_is_a_noop():
    task = portfolio_at_execution()
    before = dict(task["data"])
    ts._roll_back(ts.scenario_of(task), task, "нет такого", ts.BY_MANUAL)
    assert task["stage"] == ts.STAGE_EXECUTION
    assert task["data"] == before
    assert task["awaiting_correction"] is False


def test_last_auto_stage_is_nearest_auto_stage_before_the_given_one():
    scenario = ts.SCENARIOS["portfolio"]
    assert ts._last_auto_stage(scenario, ts.STAGE_VALIDATION) == ts.STAGE_EXECUTION
    assert ts._last_auto_stage(scenario, ts.STAGE_PLANNING) is None
    assert ts._last_auto_stage(scenario, "нет такого") is None


def test_invariant_violations_without_claims_change_nothing():
    task = portfolio_at_validation()
    updated, label = ts.apply_invariant_violations(task, [])
    assert label is None
    assert updated == task
    assert updated is not task


def test_invariant_violations_roll_back_to_work_stage_keeping_the_result():
    task = portfolio_at_validation()
    updated, label = ts.apply_invariant_violations(task, _violations())
    assert label == "проверка → формирование портфеля"
    assert updated["stage"] == ts.STAGE_EXECUTION
    assert updated["data"]["allocation"] == task["data"]["allocation"]
    assert updated["awaiting_correction"] is True
    assert updated[ts.INVARIANT_VIOLATIONS_KEY] == _violations()
    assert updated["transitions"][-1]["by"] == ts.BY_INVARIANTS
    assert "№2" in updated["transitions"][-1]["note"]
    assert task["stage"] == ts.STAGE_VALIDATION  # исходная задача не изменена


def test_invariant_violations_before_any_work_stage_keep_state_but_store_claims():
    task = ts.new_task("portfolio", "g")
    updated, label = ts.apply_invariant_violations(task, _violations())
    assert label is None
    assert updated["stage"] == ts.STAGE_PLANNING
    assert updated[ts.INVARIANT_VIOLATIONS_KEY] == _violations()
    assert updated["transitions"] == []


def test_state_response_stores_trimmed_step_and_expected_action_and_ignores_blank():
    task = ts.new_task("asset", "g")
    change = ts.apply_state_response(
        task, {"step": "  сбор вводных ", "expected_action": "назвать горизонт"}
    )
    assert change.task["step"] == "сбор вводных"
    assert change.task["expected_action"] == "назвать горизонт"
    again = ts.apply_state_response(change.task, {"step": "   ", "expected_action": 5})
    assert again.task["step"] == "сбор вводных"
    assert again.task["expected_action"] == "назвать горизонт"


def test_parse_start_response_cases():
    assert ts.parse_start_response({"task_type": "asset", "goal": "  разбор ОФЗ "}) == (
        "asset",
        "разбор ОФЗ",
    )
    assert ts.parse_start_response({"task_type": "asset", "goal": ""}) == (
        "asset",
        ts.SCENARIOS["asset"].label,
    )
    assert ts.parse_start_response({"task_type": "review", "goal": 5}) == (
        "review",
        ts.SCENARIOS["review"].label,
    )
    assert ts.parse_start_response({"task_type": "none"}) is None
    assert ts.parse_start_response({"task_type": None}) is None
    assert ts.parse_start_response({"task_type": ["asset"]}) is None
    assert ts.parse_start_response({}) is None


def test_render_exchanges_labels_roles():
    text = ts.render_exchanges(
        [
            {"role": "user", "content": "Вопрос"},
            {"role": "assistant", "content": "Ответ"},
            {"role": "user"},
        ]
    )
    assert text == "Пользователь: Вопрос\nАгент: Ответ\nПользователь: "
    assert ts.render_exchanges([]) == ""


def test_context_message_of_fresh_task_names_stage_next_stage_and_missing():
    text = ts.build_context_message(ts.new_task("portfolio", "пенсия"))
    assert "Составление портфеля — «пенсия»" in text
    assert "Этап 1 из 4: планирование." in text
    assert "Следующий этап: формирование портфеля." in text
    assert "Не хватает для перехода" in text
    for key in ("goal_type", "horizon", "risk", "constraints"):
        assert f"- {ts.key_label(key)}" in text
    assert "Уже собрано" not in text


def test_context_message_has_step_expected_action_and_collected_data():
    task = ts.new_task("portfolio", "пенсия")
    task["step"] = "сбор вводных"
    task["expected_action"] = "назвать горизонт"
    task["data"] = {"goal_type": "пенсия", "risk": ""}
    text = ts.build_context_message(task)
    assert "Текущий шаг: сбор вводных" in text
    assert "Ожидаемое действие: назвать горизонт" in text
    assert "Уже собрано:\n- цель накоплений: пенсия" in text
    assert "отношение к риску: " not in text  # пустые значения не показываются


def test_context_message_asks_for_direct_answer_when_only_gate_is_missing():
    task = apply(ts.new_task("portfolio", "g"), PORTFOLIO_INPUTS).task
    task = apply(task, {ts.BRIEF_KEY: "сводка"}).task
    text = ts.build_context_message(task)
    assert "Осталось только решение пользователя" in text
    assert ts.key_label(ts.BRIEF_VERDICT_KEY) in text


def test_context_message_of_paused_task_stops_after_pause_note():
    task = portfolio_at_execution()
    task["paused"] = True
    task["step"] = "не должен попасть"
    text = ts.build_context_message(task)
    assert "ЗАДАЧА НА ПАУЗЕ" in text
    assert "не должен попасть" not in text
    assert "Не хватает" not in text


def test_context_message_after_invariant_rollback_quotes_the_same_claims():
    updated, _ = ts.apply_invariant_violations(portfolio_at_validation(), _violations())
    text = ts.build_context_message(updated)
    assert "НАРУШИЛ ИНВАРИАНТЫ" in text
    assert "- инвариант №2 (без криптовалют): в структуре есть токены" in text
    assert "Этап не сменится, пока не появится новый результат" in text


def test_context_message_while_waiting_for_correction_and_when_all_collected():
    task = portfolio_at_execution()
    task["data"]["allocation"] = "акции 50%, облигации 50%"
    assert "Все пункты этого этапа собраны" in ts.build_context_message(task)
    task["awaiting_correction"] = True
    text = ts.build_context_message(task)
    assert "вернулся на этот этап" in text
    assert "Все пункты" not in text


def test_context_message_of_last_stage_has_no_next_stage():
    task = portfolio_at_validation()
    task["stage"] = ts.STAGE_DONE
    text = ts.build_context_message(task)
    assert "Это последний этап." in text
    assert "Все пункты" not in text


def test_state_user_content_describes_scenario_stage_keys_gate_and_exchanges():
    task = ts.new_task("portfolio", "пенсия")
    messages = [{"role": "user", "content": "Хочу накопить"}]
    text = ts.build_state_user_content(task, messages)
    assert "Сценарий задачи: Составление портфеля." in text
    assert "Текущий этап: planning." in text
    assert "Допустимые следующие этапы: execution." in text
    assert f"Условие-гейт этапа: ключ {ts.BRIEF_VERDICT_KEY}" in text
    assert '"confirmed"' in text
    assert "заполняются только после того" in text
    assert f"- goal_type: {ts.key_label('goal_type')}" in text
    assert "- preferred_assets:" in text
    assert "Пользователь: Хочу накопить" in text
    for stage in ts.SCENARIOS["portfolio"].stages:
        assert f"- {stage.name} ({stage.label})" in text


def test_state_user_content_of_stage_without_gate_has_no_gate_note():
    text = ts.build_state_user_content(ts.new_task("asset", "g"), [])
    assert "Условие-гейт" not in text
    assert "заполняются только после" not in text


def test_state_user_content_of_finished_task_says_nothing_to_fill():
    task = portfolio_at_validation()
    task["stage"] = ts.STAGE_DONE
    text = ts.build_state_user_content(task, [])
    assert "нет (задача завершена)" in text
    assert "- (нет)" in text
    assert "Обязательные для перехода ключи текущего этапа: нет." in text


def test_start_user_content_lists_every_scenario_and_exchanges():
    text = ts.build_start_user_content([{"role": "user", "content": "Разбери ОФЗ"}])
    for key, scenario in ts.SCENARIOS.items():
        assert f"- {key}: {scenario.label}" in text
    assert "Пользователь: Разбери ОФЗ" in text


def test_state_line_variants():
    paused = ts.new_task("portfolio", "g")
    paused["paused"] = True
    assert ts.state_line(paused) == "⏸ Составление портфеля · планирование · задача на паузе"

    fresh = ts.state_line(ts.new_task("portfolio", "g"))
    assert fresh.startswith("📍 Составление портфеля · планирование · ждём: ")
    assert ts.key_label("goal_type") in fresh

    done = portfolio_at_validation()
    done["stage"] = ts.STAGE_DONE
    assert ts.state_line(done).endswith("задача завершена, очистить — /smart_agent_task_done")

    violated, _ = ts.apply_invariant_violations(portfolio_at_validation(), _violations())
    assert "нарушены инварианты №2 — нужен новый вариант" in ts.state_line(violated)

    complete = portfolio_at_execution()
    complete["data"]["allocation"] = "акции 50%, облигации 50%"
    assert ts.state_line(complete).endswith("все пункты этапа собраны")
    complete["expected_action"] = "подтвердить структуру"
    assert ts.state_line(complete).endswith("ждём: подтвердить структуру")
    complete["awaiting_correction"] = True
    assert ts.state_line(complete).endswith("ждём правку: подтвердить структуру")
    complete["expected_action"] = ""
    assert ts.state_line(complete).endswith("ждём правку: что именно изменить")


def test_state_line_truncates_a_long_tail_to_120_characters():
    task = portfolio_at_execution()
    task["data"]["allocation"] = "акции 50%, облигации 50%"
    task["expected_action"] = "очень длинное ожидание " * 20
    line = ts.state_line(task)
    tail = line.split(" · ", 2)[2]
    assert len(tail) == 120
    assert tail.endswith("...")


def test_describe_state_of_fresh_task():
    text = ts.describe_state(ts.new_task("portfolio", "пенсия"))
    assert "📋 Задача: Составление портфеля" in text
    assert "Цель: пенсия" in text
    assert "Этап: планирование (1 из 4)" in text
    assert "Текущий шаг: —" in text
    assert "Ожидаемое действие: —" in text
    assert "Собрано:\n— пока ничего" in text
    assert "Не хватает до следующего этапа:\n- цель накоплений" in text
    assert "История переходов" not in text


def test_describe_state_marks_pause_with_and_without_time():
    task = ts.new_task("asset", "")
    task["paused"] = True
    assert "Цель: —" in ts.describe_state(task)
    assert "· ⏸ на паузе" in ts.describe_state(task)
    assert "на паузе с " not in ts.describe_state(task)
    task["paused_at"] = "2026-10-01T12:00:00"
    assert "на паузе с 2026-10-01T12:00:00" in ts.describe_state(task)


def test_describe_state_after_invariant_rollback_and_with_history():
    updated, _ = ts.apply_invariant_violations(portfolio_at_validation(), _violations())
    text = ts.describe_state(updated)
    assert "⛔ Результат вернулся на доработку — нарушены инварианты:" in text
    assert "- инвариант №2 (без криптовалют): в структуре есть токены" in text
    assert "Ждём новый вариант результата" in text
    assert "История переходов:" in text
    assert "инварианты" in text.split("История переходов:")[1]


def test_describe_state_waiting_for_correction_complete_and_done():
    task = portfolio_at_execution()
    task["data"]["allocation"] = "акции 50%, облигации 50%"
    assert "Не хватает до следующего этапа: ничего, все пункты собраны." in ts.describe_state(
        task
    )
    task["awaiting_correction"] = True
    assert "Ждём правку: задача вернулась на этот этап" in ts.describe_state(task)
    task["awaiting_correction"] = False
    task["stage"] = ts.STAGE_DONE
    assert "Не хватает до следующего этапа" not in ts.describe_state(task)


def test_render_transitions_formats_moves_rejections_and_unknown_authors():
    task = ts.new_task("portfolio", "g")
    task["transitions"] = [
        {"from": "planning", "to": "execution", "by": ts.BY_AUTO, "note": ""},
        {"from": "execution", "to": "execution", "by": ts.BY_REJECTED, "note": "причина"},
        {"from": "planning", "to": "planning", "by": ts.BY_GATE_REJECTED, "note": ""},
        {"from": "execution", "to": "validation", "by": "чужой", "note": "заметка"},
        {"from": "нет", "to": "тоже нет", "by": ts.BY_MANUAL},
    ]
    assert ts.render_transitions(task).split("\n") == [
        "- планирование → формирование портфеля · автомат",
        "- формирование портфеля · отклонено — причина",
        "- планирование · сводка не подтверждена",
        "- формирование портфеля → проверка · чужой — заметка",
        "- нет → тоже нет · вручную",
    ]
    assert ts.render_transitions(ts.new_task("asset", "g")) == ""


def test_transitions_of_broken_field_is_empty_list():
    assert ts.transitions({"transitions": "x"}) == []
    assert ts.invariant_violations({ts.INVARIANT_VIOLATIONS_KEY: "x"}) == []


class _FixedDateTime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 1, 12, 0, 0)


def test_archive_fact_is_one_line_with_date_goal_and_results(monkeypatch):
    monkeypatch.setattr(ts, "datetime", _FixedDateTime)
    task = portfolio_at_validation()
    task["goal"] = "портфель на пенсию"
    fact = ts.archive_fact(task)
    assert fact == (
        "2026-10-01: задача «Составление портфеля» завершена. Цель: портфель на пенсию."
        " Итог — структура портфеля в процентах: акции 40%, облигации 60%"
    )
    assert "\n" not in fact


def test_archive_fact_without_results_or_goal_and_with_truncation(monkeypatch):
    monkeypatch.setattr(ts, "datetime", _FixedDateTime)
    bare = ts.archive_fact(ts.new_task("asset", ""))
    assert bare == "2026-10-01: задача «Разбор актива или идеи» завершена. Цель: —."

    long_task = ts.new_task("asset", "g")
    long_task["data"]["analysis"] = "разбор " * 200
    fact = ts.archive_fact(long_task)
    assert len(fact) == 600
    assert fact.endswith("...")


def test_new_task_clamps_negative_processed_pairs_and_starts_clean():
    task = ts.new_task("review", "ревизия", processed_pairs=-3)
    assert task["processed_pairs"] == 0
    assert task["stage"] == ts.STAGE_PLANNING
    assert task["data"] == {}
    assert task["transitions"] == []
    assert ts.new_task("review", "g", processed_pairs=4)["processed_pairs"] == 4
