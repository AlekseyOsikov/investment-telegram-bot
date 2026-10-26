"""Чистые правила execution loop (День 5): разбор трекера, промпт, классификация, метрики.

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
FAIL_BAD_CHECK = "некорректная приёмка"
FAIL_BUDGET = "превышен бюджет попытки"

# Коды bash: 126 — файл не исполняется, 127 — команда или файл не найдены.
BROKEN_CHECK_CODES = (126, 127)
BUDGET_SUBTYPE = "error_max_budget_usd"

_TASK_LINE_RE = re.compile(r"^- \[( |x|X|!)\] +(.+?)\s*$")
_ID_RE = re.compile(r"^([A-Za-zА-Яа-я]{1,8}-?\d+):?$")
_CHECK_LINE_RE = re.compile(r"^\s*Проверка:\s*(.+?)\s*$")
_BACKTICK_RE = re.compile(r"`([^`]+)`")

STATUS_OPEN = "open"
STATUS_DONE = "done"
STATUS_BLOCKED = "blocked"
_MARK_TO_STATUS = {" ": STATUS_OPEN, "x": STATUS_DONE, "X": STATUS_DONE, "!": STATUS_BLOCKED}


@dataclass(frozen=True)
class Task:
    """Задача трекера: `- [ ] T01 заголовок` и отступленное описание под ней."""

    task_id: str
    title: str
    body: str
    status: str
    checks: tuple[str, ...]  # команды из строк «Проверка: `cmd`» (необязательны)
    line: int  # номер строки с чекбоксом (с 0)


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
    api_error: str = ""  # ассистент не смог начать работу: ошибка авторизации/API/запуска
    broken_check: bool = False  # команда «Проверка:» не нашлась (код 126/127): повтор бессмыслен
    budget_exceeded: bool = False  # claude остановлен лимитом --max-budget-usd


@dataclass
class TaskResult:
    task_id: str
    attempts: list[Attempt] = field(default_factory=list)
    passed: bool = False
    commit_by_runner: bool = False  # ассистент не закоммитил сам — коммит сделал раннер
    skipped: str = ""  # причина, по которой модель не запускалась (например, FAIL_BAD_CHECK)
    note: str = ""  # аномалии: взял не ту задачу, отметил несколько и т. п.
    branch: str = ""  # ветка задачи (режим commits=branches)

    @property
    def seconds(self) -> float:
        return sum(a.seconds for a in self.attempts)

    @property
    def first_try(self) -> bool:
        return self.passed and len(self.attempts) == 1 and not self.commit_by_runner

    @property
    def failure_class(self) -> str | None:
        if self.passed:
            return None
        return self.skipped or classify_failure(self.attempts)


def _split_id(head: str, position: int) -> tuple[str, str]:
    """`T01 заголовок` → (`T01`, `заголовок`); без id в начале — `№<позиция>` и вся строка."""
    first, _, rest = head.partition(" ")
    if _ID_RE.match(first) and rest.strip():
        return first.rstrip(":"), rest.strip()
    return f"№{position}", head.strip()


def parse_tracker(text: str) -> list[Task]:
    """Markdown-список `- [ ]` / `- [x]` / `- [!]` → задачи в порядке файла.

    Описание — строки под задачей с отступом (и пустые между ними); любая неотступленная строка
    заканчивает задачу. Строка «Проверка: `команда`» в описании — приёмочная команда задачи.
    Одинаковые id в одном трекере — ValueError (по id раннер сравнивает состояния).
    """
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _TASK_LINE_RE.match(line)]
    tasks: list[Task] = []
    for n, i in enumerate(starts):
        mark, head = _TASK_LINE_RE.match(lines[i]).groups()
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body_lines: list[str] = []
        for line in lines[i + 1 : end]:
            if line.strip() and not line[0].isspace():
                break
            body_lines.append(line.strip())
        body = "\n".join(body_lines).strip()
        checks: list[str] = []
        for line in body_lines:
            found = _CHECK_LINE_RE.match(line)
            if found:
                commands = _BACKTICK_RE.findall(found.group(1))
                checks += commands or [found.group(1)]
        task_id, title = _split_id(head, n + 1)
        tasks.append(
            Task(task_id, title, body, _MARK_TO_STATUS[mark], tuple(checks), i)
        )
    ids = [t.task_id for t in tasks]
    duplicates = sorted({x for x in ids if ids.count(x) > 1})
    if duplicates:
        raise ValueError(f"повторяются id задач: {', '.join(duplicates)}")
    return tasks


def first_open(tasks: list[Task]) -> Task | None:
    return next((t for t in tasks if t.status == STATUS_OPEN), None)


def newly_done(before: list[Task], after: list[Task]) -> list[Task]:
    """Задачи, которые были открыты до попытки и отмечены `[x]` после неё."""
    was_open = {t.task_id for t in before if t.status == STATUS_OPEN}
    return [t for t in after if t.status == STATUS_DONE and t.task_id in was_open]


COMMITS_SEQUENTIAL = "sequential"  # все задачи друг за другом в одной ветке
COMMITS_BRANCHES = "branches"  # каждая задача в своей ветке от общей точки старта
COMMIT_MODES = (COMMITS_SEQUENTIAL, COMMITS_BRANCHES)


def task_branch_name(
    base_branch: str, task_id: str, taken: set[str] | frozenset[str] = frozenset()
) -> str:
    """Имя ветки задачи: `<ветка прогона>-<id>`; `№3` → `N3`; занятое имя получает `-2`, `-3`…

    Слэш не используется: ветка `run1` и ветка `run1/T01` в git несовместимы.
    """
    safe = re.sub(r"[^\w.-]+", "-", task_id.replace("№", "N")).strip("-.") or "task"
    name = f"{base_branch}-{safe}"
    candidate, n = name, 1
    while candidate in taken:
        n += 1
        candidate = f"{name}-{n}"
    return candidate


def mark_done(text: str, tasks: list[Task]) -> str:
    """Меняет чекбоксы переданных задач на `[x]` (в режиме branches трекер основной ветки ведёт
    раннер: коммит задачи лежит в её ветке, а здесь остаётся только отметка)."""
    lines = text.split("\n")
    for task in tasks:
        lines[task.line] = lines[task.line].replace("[ ]", "[x]", 1)
    return "\n".join(lines)


def mark_blocked(text: str, task: Task) -> str:
    """Меняет чекбокс задачи на `[!]` (раннер так исключает задачу после провала всех попыток)."""
    lines = text.split("\n")
    lines[task.line] = lines[task.line].replace("[ ]", "[!]", 1)
    return "\n".join(lines)


PROMPT_TEMPLATE = """Ты работаешь в автономном цикле: человека рядом нет, вопросов не задавай, \
решения принимай сам. Список задач — файл {tracker}.

Порядок:
1. Открой {tracker} и возьми ПЕРВУЮ невыполненную задачу («- [ ]»); задачи «- [x]» уже сделаны, \
«- [!]» пропусти. Другие задачи не трогай.
2. Выбери профиль по типу задачи: баг — субагент bug-fix; недостающие тесты — test-writer; \
расхождения в документации — doc-sync; вопрос по коду — research; фичу и рефакторинг делай сам. \
Работай по правилам проекта (CLAUDE.md и документам, на которые он ссылается).
3. Выполни задачу, правя только то, что она требует. Тесты не ослабляй и не удаляй.
4. Если у задачи есть строка «Проверка:», выполни указанную команду и добейся успеха; \
прогони также проверки проекта (сборка, линтер, тесты).
5. Отметь задачу в {tracker} как выполненную («- [x]») и закоммить свои изменения вместе с \
этой отметкой одним коммитом «<ID задачи>: <кратко>» (git add нужных файлов, без push).
6. Ответь одной строкой: что сделано.
"""

RETRY_SUFFIX = """
ВНИМАНИЕ: предыдущая попытка не прошла приёмку. Её изменения откачены, рабочая копия чистая. \
Результат приёмки предыдущей попытки:
{check_output}
"""


def build_prompt(tracker: str, previous_check_output: str | None = None) -> str:
    """Один и тот же текст для каждой задачи: конкретную задачу ассистент выбирает сам."""
    prompt = PROMPT_TEMPLATE.format(tracker=tracker)
    if previous_check_output is not None:
        tail = previous_check_output.strip()[-1500:]
        prompt += RETRY_SUFFIX.format(check_output=tail or "(пусто)")
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


def is_broken_check(returncode: int) -> bool:
    """Команда приёмки не запустилась (файл или команда не найдены): задача тут ни при чём."""
    return returncode in BROKEN_CHECK_CODES


def is_budget_exceeded(data: dict) -> bool:
    """Итоговое событие `claude`: остановлен лимитом `--max-budget-usd`."""
    return data.get("subtype") == BUDGET_SUBTYPE


def api_error_text(data: dict, returncode: int, raw: str) -> str:
    """Текст ошибки, если `claude` не смог начать работу (авторизация, 403/5xx, лимиты, нет вывода).

    Признак: итоговое событие с `is_error` и не больше одного хода, либо код выхода ≠ 0 без
    разобранного итога. Ошибка посреди работы (много ходов, таймаут, бюджет) — это не он, а
    провал задачи.
    """
    if is_budget_exceeded(data):
        return ""
    if data.get("is_error") and (data.get("num_turns") or 0) <= 1:
        return str(data.get("result") or "ошибка без текста")[:300]
    if returncode not in (0, -9) and not data:
        return raw.strip()[-300:] or f"код выхода {returncode}"
    return ""


def classify_failure(attempts: list[Attempt]) -> str:
    """Причина провала по последней попытке (эвристика; ручная проверка лога обязательна)."""
    if not attempts:
        return FAIL_TOOLING
    last = attempts[-1]
    if last.api_error:
        return FAIL_TOOLING
    if last.broken_check:
        return FAIL_BAD_CHECK
    if last.budget_exceeded:
        return FAIL_BUDGET
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
    streak_minutes: float  # время работы ассистента в выполненных подряд задачах, минут
    first_failure: str | None
    first_failure_class: str | None
    mean_seconds: float
    total_cost_usd: float


def compute_metrics(results: list[TaskResult]) -> Metrics:
    streak = strict = 0
    streak_seconds = 0.0
    for r in results:
        if not r.passed:
            break
        streak += 1
        streak_seconds += r.seconds
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
        streak_minutes=streak_seconds / 60,
        first_failure=failed.task_id if failed else None,
        first_failure_class=failed.failure_class if failed else None,
        mean_seconds=total_seconds / len(results) if results else 0.0,
        total_cost_usd=cost,
    )


def _fmt_seconds(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60} мин {total % 60} с" if total >= 60 else f"{total} с"


def _note(result: TaskResult) -> str:
    parts = [result.note, f"ветка `{result.branch}`" if result.branch and result.passed else ""]
    return "; ".join(p for p in parts if p)


def render_log(run_name: str, model: str, results: list[TaskResult]) -> str:
    """Лог выполнения (задача → результат → время) и сводка метрик в Markdown."""
    m = compute_metrics(results)
    lines = [
        f"# Лог execution loop: {run_name}",
        "",
        f"Модель: `{model}`. Задач в прогоне: {m.total}.",
        "",
        "| Задача | Результат | Попыток | Время | Ходов | Файлов | Причина провала | Примечание |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        verdict = "✅ с 1-й попытки" if r.first_try else ("✅ со 2-й+" if r.passed else "❌")
        if r.commit_by_runner:
            verdict += " (коммит раннера)"
        turns = sum(a.turns or 0 for a in r.attempts) or "—"
        changed = r.attempts[-1].changed_files if r.attempts else 0
        lines.append(
            f"| {r.task_id} | {verdict} | {len(r.attempts)} | {_fmt_seconds(r.seconds)} | "
            f"{turns} | {changed} | {r.failure_class or ''} | {_note(r)} |"
        )
    pct = 100 * m.first_try / m.total if m.total else 0
    lines += [
        "",
        "## Метрики",
        "",
        f"- Выполнено задач: {m.passed} из {m.total}.",
        f"- Подряд без паузы с начала пула: **{m.streak}** (с первой попытки: {m.strict_streak}).",
        f"- Время без пауз: **{m.streak_minutes:.1f} мин** (работа ассистента в задачах серии).",
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
