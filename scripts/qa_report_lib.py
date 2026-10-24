"""Чистые правила QA-отчёта по PR (`scripts/qa_report.py`): разбор итогов проверок и сборка одного
комментария. Не знает про GitHub, сеть и диск: файлы результатов читает CLI, сюда приходит текст.

Источник данных — каталог результатов `scripts/qa_ci.sh`: `status.txt` (строки `ключ=число`:
код выхода каждой проверки и счётчики ruff), `pytest.txt` (вывод pytest с покрытием) и
`smoke/report.md` (отчёт `scripts/smoke.py`). Текст из вывода тестов порождён кодом PR, поэтому в
комментарий он попадает только обезвреженным (`sanitize_text`) и в блоке кода.

Тесты — `tests/test_qa_report_lib.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def sanitize_text(text: str) -> str:
    """Обезвреживает текст из вывода тестов (его пишет код PR): `@user` не уведомляет людей,
    HTML-комментарии (в том числе поддельный маркер отчёта) не работают. Та же защита, что в
    AI-ревью (`pr_review_lib.sanitize_text`); копия, чтобы модуль не тянул `src/` (CLI запускается
    без него)."""
    return text.replace("<!--", "&lt;!--").replace("@", "@\u200b")


# Маркер комментария: по нему и по автору (github-actions[bot]) комментарий находят для обновления.
MARKER = "<!-- pr-qa-report -->"
MAX_COMMENT_CHARS = 60000

MAX_FAILED_TESTS_SHOWN = 15
MAX_FAILURES_SHOWN = 15
MAX_LINE_CHARS = 300

# Фиксированная оговорка: входит в КАЖДЫЙ комментарий (тест это проверяет).
DISCLAIMER = (
    "Отчёт автоматический: код-тесты и smoke выполнены на GitHub-hosted runner без модели, сети и "
    "Telegram. Он не заменяет ручные проверки (`/watch`, `/support`, ответы живой модели) и не "
    "является ревью кода. Скриншоты шагов smoke — в артефакте прогона «qa-report»."
)

CHECK_KEYS = ("compile", "ruff", "pytest", "smoke")


@dataclass(frozen=True)
class SmokeFailure:
    scenario: str
    step: str
    problem: str
    where: str


@dataclass(frozen=True)
class SmokeSummary:
    passed: int
    total: int
    failures: tuple[SmokeFailure, ...]


@dataclass(frozen=True)
class QaInput:
    sha: str
    status: dict[str, int]
    pytest_text: str
    smoke_report: str | None
    changed: tuple[str, ...] = ()
    run_url: str | None = None


# --------------------------------------------------------------------------- #
# Разбор
# --------------------------------------------------------------------------- #


def parse_status(text: str) -> dict[str, int]:
    """`status.txt`: строки `ключ=число`; всё остальное игнорируется."""
    result: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key and value.lstrip("-").isdigit():
            result[key] = int(value)
    return result


_PYTEST_COUNT_RE = re.compile(r"\b(\d+) (passed|failed|errors?|skipped)\b")


def parse_pytest_counts(text: str) -> dict[str, int]:
    """Итоговая строка pytest («2 failed, 1835 passed in 8.7s») → {'failed': 2, 'passed': 1835}.
    Пустой словарь, если итоговой строки нет (pytest не дошёл до итога)."""
    for line in reversed(text.splitlines()):
        if " in " not in line:
            continue
        pairs = _PYTEST_COUNT_RE.findall(line)
        if pairs:
            counts: dict[str, int] = {}
            for number, word in pairs:
                key = "error" if word.startswith("error") else word
                counts[key] = counts.get(key, 0) + int(number)
            return counts
    return {}


_COVERAGE_RE = re.compile(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", re.MULTILINE)


def parse_coverage_total(text: str) -> int | None:
    match = _COVERAGE_RE.search(text)
    return int(match.group(1)) if match else None


_FAILED_LINE_RE = re.compile(r"^(?:FAILED|ERROR) \S+")


def failed_test_lines(text: str, limit: int = MAX_FAILED_TESTS_SHOWN) -> list[str]:
    """Строки краткого итога pytest `FAILED tests/...::test - причина` (не больше `limit`)."""
    lines = [line.strip()[:MAX_LINE_CHARS] for line in text.splitlines()]
    return [line for line in lines if _FAILED_LINE_RE.match(line)][:limit]


_SMOKE_SUMMARY_RE = re.compile(r"Сценариев пройдено: (\d+) из (\d+)")
_SMOKE_SCENARIO_RE = re.compile(r"^## (.+)$")
_SMOKE_STEP_FAILED_RE = re.compile(r"^\d+\. ❌ \*\*(.+?)\*\*")
_SMOKE_STEP_OK_RE = re.compile(r"^\d+\. ✅")
_SMOKE_PROBLEM_RE = re.compile(r"^\s+- Что сломалось: (.+)$")
_SMOKE_WHERE_RE = re.compile(r"^\s+- Где искать: (.+)$")


def parse_smoke_report(text: str) -> SmokeSummary | None:
    """Отчёт `smoke.py` → число сценариев и упавшие шаги; None, если это не отчёт smoke."""
    summary = _SMOKE_SUMMARY_RE.search(text)
    if summary is None:
        return None
    failures: list[SmokeFailure] = []
    scenario = ""
    step = ""
    problem = ""
    where = ""

    def flush() -> None:
        nonlocal step, problem, where
        if step:
            failures.append(SmokeFailure(scenario, step, problem, where))
        step, problem, where = "", "", ""

    for line in text.splitlines():
        if match := _SMOKE_SCENARIO_RE.match(line):
            flush()
            scenario = match.group(1)
        elif match := _SMOKE_STEP_FAILED_RE.match(line):
            flush()
            step = match.group(1)
        elif step and (match := _SMOKE_PROBLEM_RE.match(line)):
            problem = f"{problem}; {match.group(1)}" if problem else match.group(1)
        elif step and (match := _SMOKE_WHERE_RE.match(line)):
            where = match.group(1)
        elif _SMOKE_STEP_OK_RE.match(line):
            flush()
    flush()
    return SmokeSummary(int(summary.group(1)), int(summary.group(2)), tuple(failures))


# --------------------------------------------------------------------------- #
# Итог проверок
# --------------------------------------------------------------------------- #


def check_ok(data: QaInput, key: str) -> bool:
    """Проверка прошла: код выхода 0. Нет записи в `status.txt` — проверка не выполнялась."""
    return data.status.get(key) == 0


def overall_ok(data: QaInput) -> bool:
    return all(check_ok(data, key) for key in CHECK_KEYS)


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _ruff_cell(data: QaInput) -> str:
    ok = check_ok(data, "ruff")
    before, after = data.status.get("ruff_before"), data.status.get("ruff_after")
    if before is None or after is None:
        return f"{_mark(ok)} {'нет данных' if not ok else 'ок'}"
    note = "не выросло" if after <= before else "ВЫРОСЛО"
    return f"{_mark(ok)} замечаний в изменённых файлах: {before} → {after} ({note})"


def _pytest_cell(data: QaInput) -> str:
    ok = check_ok(data, "pytest")
    counts = parse_pytest_counts(data.pytest_text)
    if not counts:
        return f"{_mark(ok)} итог pytest не получен"
    parts = [f"{number} {word}" for word, number in counts.items()]
    return f"{_mark(ok)} {', '.join(parts)}"


def _smoke_cell(data: QaInput, summary: SmokeSummary | None) -> str:
    ok = check_ok(data, "smoke")
    if summary is None:
        return f"{_mark(False)} отчёт smoke не создан"
    return f"{_mark(ok)} сценариев {summary.passed} из {summary.total}"


def _fence(text: str) -> str:
    longest = max((len(found) for found in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _truncate(body: str, limit: int) -> str:
    if len(body) <= limit:
        return body
    note = "\n\n_… текст усечён по лимиту длины._"
    return body[: max(0, limit - len(note))] + note


def build_comment(data: QaInput, limit: int = MAX_COMMENT_CHARS) -> str:
    """Один комментарий с итогами обоих уровней; оговорка — в каждом варианте."""
    summary = parse_smoke_report(data.smoke_report) if data.smoke_report else None
    ok = overall_ok(data) and summary is not None
    coverage = parse_coverage_total(data.pytest_text)

    sha = data.sha[:7] if data.sha else "—"
    run = f" · [прогон]({data.run_url})" if data.run_url else ""
    lines = [
        MARKER,
        f"## QA-отчёт: {_mark(ok)} {'всё прошло' if ok else 'есть замечания'}",
        "",
        f"Коммит `{sha}`{run}. Изменённых файлов: {len(data.changed)}.",
        "",
        "| Проверка | Результат |",
        "|---|---|",
        f"| Синтаксис (`compileall src`) | {_mark(check_ok(data, 'compile'))} |",
        f"| Линтер (`ruff`) | {_ruff_cell(data)} |",
        f"| Тесты (`pytest`) | {_pytest_cell(data)} |",
        f"| Покрытие `src/` и `scripts/` | {f'{coverage}%' if coverage is not None else '—'} |",
        f"| Smoke-сценарии | {_smoke_cell(data, summary)} |",
    ]

    failed_tests = failed_test_lines(data.pytest_text)
    if failed_tests:
        text = sanitize_text("\n".join(failed_tests))
        fence = _fence(text)
        lines += ["", "### Упавшие тесты", "", f"{fence}text", text, fence]

    if summary is not None and summary.failures:
        lines += ["", "### Упавшие шаги smoke", ""]
        for item in summary.failures[:MAX_FAILURES_SHOWN]:
            lines.append(
                f"- **{sanitize_text(item.scenario)}** → «{sanitize_text(item.step)}»: "
                f"{sanitize_text(item.problem) or 'без пояснения'}"
            )
            if item.where:
                lines.append(f"  - где искать: {sanitize_text(item.where)}")
        hidden = len(summary.failures) - MAX_FAILURES_SHOWN
        if hidden > 0:
            lines.append(f"- … и ещё {hidden}.")

    tail = f"\n\n---\n{DISCLAIMER}\n"
    # Усекается тело, а не оговорка: она должна остаться в комментарии любой длины.
    return _truncate("\n".join(lines), limit - len(tail)) + tail
