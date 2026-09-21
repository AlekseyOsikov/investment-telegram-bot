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
