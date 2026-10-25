"""Тесты чистых правил execution loop (scripts/exec_loop_lib.py): разбор пула, выбор задач,
промпт, классификация сбоя, метрики и лог. Без git, процессов и сети."""

import exec_loop_lib as lib
import pytest

TRACKER = """# Трекер

Вводный текст.

- [ ] T01 bug: первая
  Симптом.
  Проверка: `pytest tests/x.py -q` и `ruff check src/x.py`
- [x] T02 test: вторая, уже сделана
- [!] T03 docs: заблокирована раннером
  Проверка: bash tasks/check.sh T03

Пояснение после списка.
- [ ] Задача без идентификатора
- [ ] 2025-1 не id, а заголовок
"""


def attempt(**kw):
    values = {"seconds": 10.0, "returncode": 0, "turns": 5, "changed_files": 1}
    values.update(kw)
    return lib.Attempt(**values)


def result(task_id, *attempts, passed=True, by_runner=False):
    return lib.TaskResult(task_id, list(attempts), passed, by_runner)

















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
    assert m.streak_minutes == pytest.approx((10 + 30) / 60)  # T01, T02; T03 прервала серию
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
    assert "Примечание" in text and "Время без пауз: **1.2 мин**" in text


def test_parse_claude_output_variants():
    event = '{"type": "result", "num_turns": 4, "total_cost_usd": 0.1}'
    assert lib.parse_claude_output(event)["num_turns"] == 4
    listing = f'[{{"type": "system"}}, {event}]'
    assert lib.parse_claude_output(listing)["total_cost_usd"] == 0.1
    assert lib.parse_claude_output("Warning: no stdin\n" + listing)["num_turns"] == 4
    assert lib.parse_claude_output("[]") == {} and lib.parse_claude_output("мусор") == {}
    assert lib.parse_claude_output("[1, 2") == {}


def test_parse_tracker_statuses_ids_titles_and_checks():
    tasks = lib.parse_tracker(TRACKER)
    assert [t.task_id for t in tasks] == ["T01", "T02", "T03", "№4", "№5"]
    assert [t.status for t in tasks] == ["open", "done", "blocked", "open", "open"]
    assert tasks[0].title == "bug: первая" and tasks[0].body.startswith("Симптом.")
    assert tasks[0].checks == ("pytest tests/x.py -q", "ruff check src/x.py")
    assert tasks[2].checks == ("bash tasks/check.sh T03",)
    assert tasks[3].title == "Задача без идентификатора" and tasks[3].checks == ()
    assert tasks[4].title == "2025-1 не id, а заголовок"


def test_task_body_stops_at_unindented_line():
    tasks = lib.parse_tracker(TRACKER)
    assert "Пояснение" not in tasks[2].body and "Вводный" not in tasks[0].body


def test_parse_tracker_empty_and_duplicate_ids():
    assert lib.parse_tracker("просто текст") == []
    with pytest.raises(ValueError, match="T01"):
        lib.parse_tracker("- [ ] T01 а\n- [ ] T01 б\n")


def test_first_open_and_newly_done():
    before = lib.parse_tracker(TRACKER)
    assert lib.first_open(before).task_id == "T01"
    after_text = TRACKER.replace("- [ ] T01", "- [x] T01").replace("- [ ] Задача", "- [x] Задача")
    done = lib.newly_done(before, lib.parse_tracker(after_text))
    assert [t.task_id for t in done] == ["T01", "№4"]
    assert lib.first_open(lib.parse_tracker("- [x] А\n- [!] Б\n")) is None
    assert lib.newly_done(before, before) == []


def test_mark_blocked_changes_only_that_checkbox():
    tasks = lib.parse_tracker(TRACKER)
    text = lib.mark_blocked(TRACKER, tasks[0])
    assert "- [!] T01 bug: первая" in text and text.count("[!]") == 2
    assert lib.first_open(lib.parse_tracker(text)).task_id == "№4"


def test_prompt_is_the_same_for_every_task_and_names_tracker():
    prompt = lib.build_prompt("tasks/tracker.md")
    assert "tasks/tracker.md" in prompt and "ПЕРВУЮ невыполненную" in prompt
    assert "предыдущая попытка" not in prompt and prompt == lib.build_prompt("tasks/tracker.md")


def test_retry_prompt_carries_only_the_tail_of_check_output():
    prompt = lib.build_prompt("t.md", "x" * 5000 + "КОНЕЦ")
    assert "предыдущая попытка" in prompt and "КОНЕЦ" in prompt and prompt.count("x") <= 1500
    assert "(пусто)" in lib.build_prompt("t.md", "  ")


def test_task_branch_name_is_flat_safe_and_unique():
    assert lib.task_branch_name("exec-loop-run1", "T05") == "exec-loop-run1-T05"
    assert lib.task_branch_name("run", "№3") == "run-N3"
    assert lib.task_branch_name("run", "A, B/C") == "run-A-B-C"
    assert lib.task_branch_name("run", "T1", {"run-T1"}) == "run-T1-2"
    assert lib.task_branch_name("run", "T1", {"run-T1", "run-T1-2"}) == "run-T1-3"
    assert "/" not in lib.task_branch_name("run", "a/b")
    assert lib.COMMIT_MODES == ("sequential", "branches")


def test_mark_done_changes_only_given_tasks():
    tasks = lib.parse_tracker(TRACKER)
    text = lib.mark_done(TRACKER, [tasks[0], tasks[3]])
    assert "- [x] T01 bug: первая" in text and "- [x] Задача без идентификатора" in text
    assert "- [ ] 2025-1" in text and text.count("[x]") == 3
    assert lib.mark_done(TRACKER, []) == TRACKER


def test_render_log_shows_branch_only_for_passed_tasks():
    done = result("T01", attempt())
    done.branch = "run-T01"
    failed = result("T02", attempt(), passed=False)
    failed.branch = "run-T02"
    text = lib.render_log("run", "m", [done, failed])
    assert "ветка `run-T01`" in text and "run-T02" not in text
