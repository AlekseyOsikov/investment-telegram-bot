"""Тесты чистых правил execution loop (scripts/exec_loop_lib.py): разбор пула, выбор задач,
промпт, классификация сбоя, метрики и лог. Без git, процессов и сети."""

import exec_loop_lib as lib
import pytest

BACKLOG = """# Пул

шапка

## T01 — первая
Симптом.
**Проверка:** `tasks/check.sh T01`

---

## T02 — вторая
Тело второй.

## T03 — третья
Тело третьей.
"""


def attempt(**kw):
    values = {"seconds": 10.0, "returncode": 0, "turns": 5, "changed_files": 1}
    values.update(kw)
    return lib.Attempt(**values)


def result(task_id, *attempts, passed=True, by_runner=False):
    return lib.TaskResult(task_id, list(attempts), passed, by_runner)


def test_parse_backlog_splits_tasks_and_strips_separator():
    tasks = lib.parse_backlog(BACKLOG)
    assert [t.task_id for t in tasks] == ["T01", "T02", "T03"]
    assert tasks[0].title == "первая" and tasks[0].body.endswith("`tasks/check.sh T01`")
    assert "---" not in tasks[0].body and tasks[1].body == "Тело второй."


def test_select_tasks_single_range_list_and_all():
    tasks = lib.parse_backlog(BACKLOG)
    ids = lambda spec: [t.task_id for t in lib.select_tasks(tasks, spec)]  # noqa: E731
    assert ids(None) == ids("") == ["T01", "T02", "T03"]
    assert ids("T02") == ["T02"] and ids("T01-T02") == ["T01", "T02"]
    assert ids("T03,T01") == ["T01", "T03"]


@pytest.mark.parametrize("spec", ["T09", "T03-T01", "T01-T09"])
def test_select_tasks_rejects_bad_spec(spec):
    with pytest.raises(ValueError):
        lib.select_tasks(lib.parse_backlog(BACKLOG), spec)


def test_prompt_has_task_and_no_hint_on_first_attempt():
    task = lib.parse_backlog(BACKLOG)[1]
    prompt = lib.build_prompt(task)
    assert "T02" in prompt and "Тело второй." in prompt and "tasks/check.sh T02" in prompt
    assert "предыдущая попытка" not in prompt


def test_retry_prompt_carries_only_check_output_tail():
    task = lib.parse_backlog(BACKLOG)[0]
    prompt = lib.build_prompt(task, "x" * 5000 + "КОНЕЦ")
    assert "предыдущая попытка" in prompt and "КОНЕЦ" in prompt and prompt.count("x") <= 1500


def test_failure_classes():
    assert lib.classify_failure([attempt(timed_out=True)]) == lib.FAIL_LOOP
    assert lib.classify_failure([attempt(changed_files=0)]) == lib.FAIL_NOT_UNDERSTOOD
    assert lib.classify_failure([attempt()]) == lib.FAIL_BROKEN_CODE
    broken = attempt(returncode=1, turns=None, changed_files=0)
    assert lib.classify_failure([broken]) == lib.FAIL_TOOLING
    assert lib.classify_failure([]) == lib.FAIL_TOOLING


def test_first_try_requires_single_attempt_and_own_commit():
    assert result("T01", attempt()).first_try
    assert not result("T01", attempt(), attempt()).first_try
    assert not result("T01", attempt(), by_runner=True).first_try
    assert not result("T01", attempt(), passed=False).first_try


def test_metrics_streaks_and_first_failure():
    results = [
        result("T01", attempt()),
        result("T02", attempt(), attempt(seconds=20)),
        result("T03", attempt(timed_out=True), passed=False),
        result("T04", attempt(cost_usd=0.5)),
    ]
    m = lib.compute_metrics(results)
    assert (m.total, m.passed, m.first_try) == (4, 3, 2)
    assert (m.streak, m.strict_streak) == (2, 1)
    assert (m.first_failure, m.first_failure_class) == ("T03", lib.FAIL_LOOP)
    assert m.mean_seconds == pytest.approx((10 + 30 + 10 + 10) / 4)
    assert m.total_cost_usd == pytest.approx(0.5)


def test_metrics_empty_and_all_passed():
    assert lib.compute_metrics([]).mean_seconds == 0.0
    m = lib.compute_metrics([result("T01", attempt()), result("T02", attempt())])
    assert (m.streak, m.first_failure) == (2, None)


def test_render_log_has_rows_and_metrics():
    results = [result("T01", attempt(seconds=75)), result("T02", attempt(), passed=False)]
    text = lib.render_log("run1", "model-x", results)
    assert "| T01 | ✅ с 1-й попытки | 1 | 1 мин 15 с |" in text
    assert "| T02 | ❌ |" in text and "нерабочий код" in text
    assert "Подряд без паузы с начала пула: **1**" in text and "model-x" in text


def test_parse_claude_output_variants():
    event = '{"type": "result", "num_turns": 4, "total_cost_usd": 0.1}'
    assert lib.parse_claude_output(event)["num_turns"] == 4
    listing = f'[{{"type": "system"}}, {event}]'
    assert lib.parse_claude_output(listing)["total_cost_usd"] == 0.1
    assert lib.parse_claude_output("Warning: no stdin\n" + listing)["num_turns"] == 4
    assert lib.parse_claude_output("[]") == {} and lib.parse_claude_output("мусор") == {}
    assert lib.parse_claude_output("[1, 2") == {}


def test_parse_backlog_stops_at_next_pool_heading():
    text = BACKLOG + "\n---\n\n# Новый пул\n\nтаблица\n\n## T04 — четвёртая\nТело.\n"
    tasks = lib.parse_backlog(text)
    assert [t.task_id for t in tasks] == ["T01", "T02", "T03", "T04"]
    assert tasks[2].body == "Тело третьей."
