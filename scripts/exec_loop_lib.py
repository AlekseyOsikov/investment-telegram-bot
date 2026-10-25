"""Чистые правила execution loop (День 5): разбор пула задач, промпт, классификация сбоя, метрики.

Оснастка вне бота: `src/` её не импортирует. Модуль не знает про git, процессы и сеть:
данные в данные; запуск и ввод-вывод — `scripts/exec_loop.py`.
Тесты — `tests/test_exec_loop_lib.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

FAIL_NOT_UNDERSTOOD = "не понял контекст"
FAIL_BROKEN_CODE = "нерабочий код"
FAIL_LOOP = "зациклился / таймаут"
FAIL_TOOLING = "сбой запуска ассистента"

_TASK_RE = re.compile(r"^## (T\d+) — (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Task:
    task_id: str
    title: str
    body: str


@dataclass
class Attempt:
    """Одна попытка выполнить задачу в свежей сессии ассистента."""

    seconds: float
    returncode: int
    timed_out: bool = False
    turns: int | None = None
    cost_usd: float | None = None
    changed_files: int = 0
    check_passed: bool = False
    check_output: str = ""


@dataclass
class TaskResult:
    task_id: str
    attempts: list[Attempt] = field(default_factory=list)
    passed: bool = False
    commit_by_runner: bool = False  # ассистент не закоммитил сам — коммит сделал раннер

    @property
    def seconds(self) -> float:
        return sum(a.seconds for a in self.attempts)

    @property
    def first_try(self) -> bool:
        return self.passed and len(self.attempts) == 1 and not self.commit_by_runner

    @property
    def failure_class(self) -> str | None:
        return None if self.passed else classify_failure(self.attempts)


def parse_backlog(text: str) -> list[Task]:
    """Разделы `## T01 — заголовок` → задачи; тело — текст до следующего заголовка (без `---`)."""
    matches = list(_TASK_RE.finditer(text))
    tasks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        body = re.split(r"^# ", body, maxsplit=1, flags=re.MULTILINE)[0]  # заголовок нового пула
        body = re.sub(r"\n-{3,}\s*$", "", body.strip()).strip()
        tasks.append(Task(match.group(1), match.group(2).strip(), body))
    return tasks


def select_tasks(tasks: list[Task], spec: str | None) -> list[Task]:
    """`T03`, `T03-T07`, `T01,T05` или пусто (все). Неизвестный id — ValueError."""
    if not spec:
        return list(tasks)
    ids = [t.task_id for t in tasks]
    chosen: list[str] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        low, _, high = part.partition("-")
        for bound in (low, high or low):
            if bound not in ids:
                raise ValueError(f"нет задачи {bound!r} в пуле")
        i, j = ids.index(low), ids.index(high or low)
        if i > j:
            raise ValueError(f"диапазон {part!r} задан наоборот")
        chosen += [x for x in ids[i : j + 1] if x not in chosen]
    return [t for t in tasks if t.task_id in chosen]


PROMPT_TEMPLATE = """Ты работаешь в автономном цикле: человека рядом нет, вопросов не задавай, \
решения принимай сам.

Задача {task_id} — {title}

{body}

Порядок:
1. Выбери профиль по типу задачи: баг — субагент bug-fix; недостающие тесты — test-writer; \
расхождения в документации — doc-sync; вопрос по коду — research; фичу и рефакторинг делай сам. \
Работай по CLAUDE.md и docs/code-style.md.
2. Выполни задачу, правя только то, что она требует. Тесты не ослабляй и не удаляй.
3. Запусти `tasks/check.sh {task_id}` и убедись, что итог — «СДЕЛАНО». Не читай и не правь \
tasks/check.sh.
4. Закоммить свои изменения одним коммитом с сообщением «{task_id}: <кратко>» (git add нужных \
файлов, без push и без правок вне задачи).
5. Ответь одной строкой: что сделано.
"""

RETRY_SUFFIX = """
ВНИМАНИЕ: предыдущая попытка не прошла приёмку. Её изменения откачены, рабочая копия чистая. \
Вывод `tasks/check.sh {task_id}` предыдущей попытки:
{check_output}
"""


def build_prompt(task: Task, previous_check_output: str | None = None) -> str:
    prompt = PROMPT_TEMPLATE.format(task_id=task.task_id, title=task.title, body=task.body)
    if previous_check_output is not None:
        tail = previous_check_output.strip()[-1500:]
        prompt += RETRY_SUFFIX.format(task_id=task.task_id, check_output=tail or "(пусто)")
    return prompt


def parse_claude_output(raw: str) -> dict:
    """Итоговое событие `claude -p --output-format json`: словарь или список событий (тогда —
    последнее с `type == "result"`); перед JSON допускается постороннее предупреждение. Не
    разобралось — пустой словарь."""
    starts = [i for i in (raw.find("["), raw.find("{")) if i != -1]
    if not starts:
        return {}
    try:
        data = json.loads(raw[min(starts) :])
    except ValueError:
        return {}
    if isinstance(data, list):
        events = (e for e in reversed(data) if isinstance(e, dict) and e.get("type") == "result")
        data = next(events, {})
    return data if isinstance(data, dict) else {}


def classify_failure(attempts: list[Attempt]) -> str:
    """Причина провала по последней попытке (эвристика; ручная проверка лога обязательна)."""
    if not attempts:
        return FAIL_TOOLING
    last = attempts[-1]
    if last.timed_out:
        return FAIL_LOOP
    if last.returncode != 0 and last.changed_files == 0 and last.turns is None:
        return FAIL_TOOLING
    if last.changed_files == 0:
        return FAIL_NOT_UNDERSTOOD
    return FAIL_BROKEN_CODE


@dataclass(frozen=True)
class Metrics:
    total: int
    passed: int
    first_try: int
    streak: int  # подряд выполненных с начала пула (с любой попытки)
    strict_streak: int  # подряд с первой попытки
    first_failure: str | None
    first_failure_class: str | None
    mean_seconds: float
    total_cost_usd: float


def compute_metrics(results: list[TaskResult]) -> Metrics:
    streak = strict = 0
    for r in results:
        if not r.passed:
            break
        streak += 1
    for r in results:
        if not r.first_try:
            break
        strict += 1
    failed = next((r for r in results if not r.passed), None)
    total_seconds = sum(r.seconds for r in results)
    cost = sum(a.cost_usd or 0.0 for r in results for a in r.attempts)
    return Metrics(
        total=len(results),
        passed=sum(r.passed for r in results),
        first_try=sum(r.first_try for r in results),
        streak=streak,
        strict_streak=strict,
        first_failure=failed.task_id if failed else None,
        first_failure_class=failed.failure_class if failed else None,
        mean_seconds=total_seconds / len(results) if results else 0.0,
        total_cost_usd=cost,
    )


def _fmt_seconds(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60} мин {total % 60} с" if total >= 60 else f"{total} с"


def render_log(run_name: str, model: str, results: list[TaskResult]) -> str:
    """Лог выполнения (задача → результат → время) и сводка метрик в Markdown."""
    m = compute_metrics(results)
    lines = [
        f"# Лог execution loop: {run_name}",
        "",
        f"Модель: `{model}`. Задач в прогоне: {m.total}.",
        "",
        "| Задача | Результат | Попыток | Время | Ходов | Файлов | Причина провала |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        verdict = "✅ с 1-й попытки" if r.first_try else ("✅ со 2-й+" if r.passed else "❌")
        if r.commit_by_runner:
            verdict += " (коммит раннера)"
        turns = sum(a.turns or 0 for a in r.attempts) or "—"
        changed = r.attempts[-1].changed_files if r.attempts else 0
        lines.append(
            f"| {r.task_id} | {verdict} | {len(r.attempts)} | {_fmt_seconds(r.seconds)} | "
            f"{turns} | {changed} | {r.failure_class or ''} |"
        )
    pct = 100 * m.first_try / m.total if m.total else 0
    lines += [
        "",
        "## Метрики",
        "",
        f"- Выполнено задач: {m.passed} из {m.total}.",
        f"- Подряд без паузы с начала пула: **{m.streak}** (с первой попытки: {m.strict_streak}).",
        f"- Первый провал: {m.first_failure or 'нет'}"
        + (f" — {m.first_failure_class}" if m.first_failure_class else "")
        + ".",
        f"- Среднее время на задачу: {_fmt_seconds(m.mean_seconds)}.",
        f"- С первого раза: {m.first_try} из {m.total} ({pct:.0f} %).",
        f"- Стоимость по данным ассистента: ${m.total_cost_usd:.2f}.",
        "",
        "Причина провала определена эвристикой по следам попытки; проверяйте вывод в "
        "`attempt*.json` и `check*.txt` каталога прогона.",
    ]
    return "\n".join(lines) + "\n"
