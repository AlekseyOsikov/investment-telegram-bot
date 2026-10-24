"""Тесты чистых правил QA-отчёта по PR (scripts/qa_report_lib.py): разбор итогов pytest, покрытия,
smoke-отчёта и статусов, сборка комментария (маркер, оговорка, обезвреживание текста PR, лимит).
Без GitHub и сети; тексты отчётов — вымышленные."""

import pytest
import qa_report_lib as lib

PYTEST_OK = """....
================ tests coverage ================
Name        Stmts   Miss  Cover
TOTAL       13475   5192    61%
2003 passed in 8.68s
"""

PYTEST_FAILED = """FAILED tests/test_a.py::test_one - AssertionError: assert 1 == 2
ERROR tests/test_b.py::test_two - ImportError
TOTAL       100   50    50%
2 failed, 1835 passed, 1 error in 9.1s
"""

SMOKE_OK = """# Smoke-отчёт

Дата: 2026-10-24 18:03. Сценариев пройдено: 5 из 5.

## Приветствие и справка

1. ✅ **/start** — `/start`
"""

SMOKE_FAILED = """# Smoke-отчёт

Дата: 2026-10-24 18:03. Сценариев пройдено: 3 из 5.

## Очередь вопросов пуста

1. ✅ **список** — `/queue`
2. ❌ **отмена** — `/queue_cancel`
   - Что сломалось: в ответе нет «Отменять нечего»
   - Что сломалось: ошибка в логе бота: KeyError
   - Где искать: llm_queue.commands.queue_cancel_command (src/llm_queue/commands.py:150)
3. ✅ **по номеру** — `/queue_cancel 3`

## Групповой чат игнорируется

1. ❌ **текст в группе** — `привет`
   - Что сломалось: бот должен был молчать, а ответил: «@admin привет»
"""

ALL_OK = {"compile": 0, "ruff": 0, "pytest": 0, "smoke": 0, "ruff_before": 3, "ruff_after": 3}


def data(**changes):
    values = {
        "sha": "0123456789abcdef",
        "status": dict(ALL_OK),
        "pytest_text": PYTEST_OK,
        "smoke_report": SMOKE_OK,
        "changed": ("a.py", "b.py"),
        "run_url": "https://github.com/o/r/actions/runs/1",
    }
    return lib.QaInput(**{**values, **changes})


def test_parse_status_ignores_garbage():
    text = "compile=0\nruff=1\n\nмусор\nпусто=\n=5\nruff_after=12\nneg=-1\n"
    assert lib.parse_status(text) == {"compile": 0, "ruff": 1, "ruff_after": 12, "neg": -1}


def test_pytest_counts_from_last_summary_line():
    assert lib.parse_pytest_counts(PYTEST_OK) == {"passed": 2003}
    assert lib.parse_pytest_counts(PYTEST_FAILED) == {"failed": 2, "passed": 1835, "error": 1}


def test_pytest_counts_empty_without_summary():
    assert lib.parse_pytest_counts("Traceback ...\nImportError\n") == {}


def test_coverage_total():
    assert lib.parse_coverage_total(PYTEST_OK) == 61
    assert lib.parse_coverage_total("нет покрытия") is None


def test_failed_test_lines_are_limited_and_clipped():
    text = "\n".join(f"FAILED tests/t.py::test_{i} - {'x' * 500}" for i in range(30))
    lines = lib.failed_test_lines(text)
    assert len(lines) == lib.MAX_FAILED_TESTS_SHOWN
    assert all(len(line) <= lib.MAX_LINE_CHARS for line in lines)
    assert lib.failed_test_lines("все тесты прошли") == []


def test_parse_smoke_ok():
    summary = lib.parse_smoke_report(SMOKE_OK)
    assert (summary.passed, summary.total, summary.failures) == (5, 5, ())


def test_parse_smoke_failures_with_scenario_step_problem_and_place():
    summary = lib.parse_smoke_report(SMOKE_FAILED)
    assert (summary.passed, summary.total) == (3, 5)
    first, second = summary.failures
    assert (first.scenario, first.step) == ("Очередь вопросов пуста", "отмена")
    assert "Отменять нечего" in first.problem and "KeyError" in first.problem
    assert first.where.startswith("llm_queue.commands.queue_cancel_command")
    # Упавший шаг в конце сценария не теряется при смене сценария и в конце файла.
    assert (second.scenario, second.step) == ("Групповой чат игнорируется", "текст в группе")


def test_parse_smoke_not_a_report():
    assert lib.parse_smoke_report("совсем другой файл") is None


def test_overall_ok_requires_every_check():
    assert lib.overall_ok(data())
    for key in lib.CHECK_KEYS:
        status = {**ALL_OK, key: 1}
        assert not lib.overall_ok(data(status=status)), key
        missing = {k: v for k, v in ALL_OK.items() if k != key}
        assert not lib.overall_ok(data(status=missing)), key


def test_comment_starts_with_marker_and_shows_results():
    comment = lib.build_comment(data())
    assert comment.startswith(lib.MARKER)
    assert "✅ всё прошло" in comment
    assert "`0123456`" in comment and "[прогон](https://github.com/o/r/actions/runs/1)" in comment
    assert "2003 passed" in comment and "61%" in comment
    assert "сценариев 5 из 5" in comment
    assert "3 → 3" in comment and "не выросло" in comment


def test_comment_marks_ruff_growth():
    status = {**ALL_OK, "ruff": 1, "ruff_before": 3, "ruff_after": 5}
    comment = lib.build_comment(data(status=status))
    assert "❌ есть замечания" in comment
    assert "3 → 5 (ВЫРОСЛО)" in comment


def test_comment_without_smoke_report_is_a_failure():
    comment = lib.build_comment(data(smoke_report=None))
    assert "отчёт smoke не создан" in comment
    assert "❌ есть замечания" in comment


def test_comment_lists_failures():
    status = {**ALL_OK, "pytest": 1, "smoke": 1}
    comment = lib.build_comment(
        data(status=status, pytest_text=PYTEST_FAILED, smoke_report=SMOKE_FAILED)
    )
    assert "### Упавшие тесты" in comment
    assert "FAILED tests/test_a.py::test_one" in comment
    assert "### Упавшие шаги smoke" in comment
    assert "**Очередь вопросов пуста** → «отмена»" in comment
    assert "где искать: llm_queue.commands.queue_cancel_command" in comment


def test_pr_controlled_text_is_neutralized():
    text = "FAILED tests/t.py::test_x - <!-- pr-qa-report --> @victim\n1 failed in 1s\n"
    report = SMOKE_FAILED
    comment = lib.build_comment(
        data(status={**ALL_OK, "pytest": 1}, pytest_text=text, smoke_report=report)
    )
    body = comment[len(lib.MARKER):]
    assert "<!--" not in body
    assert "@victim" not in body and "@admin" not in body


def test_fence_survives_backticks_in_failures():
    text = "FAILED tests/t.py::test_x - ```` внутри\n1 failed in 1s\n"
    comment = lib.build_comment(data(status={**ALL_OK, "pytest": 1}, pytest_text=text))
    assert "`````text" in comment


def test_disclaimer_in_every_comment_variant_and_no_guarantees():
    variants = [
        data(),
        data(status={**ALL_OK, "pytest": 1}, pytest_text=PYTEST_FAILED),
        data(smoke_report=SMOKE_FAILED, status={**ALL_OK, "smoke": 1}),
        data(smoke_report=None),
        data(status={}, pytest_text="", changed=(), run_url=None, sha=""),
    ]
    for variant in variants:
        comment = lib.build_comment(variant)
        assert comment.startswith(lib.MARKER)
        assert lib.DISCLAIMER in comment
        assert "гарантир" not in comment.casefold()


def test_long_comment_is_truncated_but_keeps_disclaimer():
    many = "\n".join(f"FAILED tests/t.py::test_{i} - {'я' * 250}" for i in range(15))
    comment = lib.build_comment(data(pytest_text=many + "\n15 failed in 1s\n"), limit=1500)
    assert len(comment) <= 1500
    assert "усечён" in comment
    assert comment.rstrip().endswith(lib.DISCLAIMER)


@pytest.mark.parametrize("sha", ["", "abc"])
def test_short_or_empty_sha_does_not_break(sha):
    comment = lib.build_comment(data(sha=sha))
    assert f"`{sha[:7] or '—'}`" in comment
