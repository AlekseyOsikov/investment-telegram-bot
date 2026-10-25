"""Execution loop (День 5): ассистент берёт задачу из трекера → выполняет → коммитит → следующая.

    python scripts/exec_loop.py --tracker tasks/tracker.md --branch exec-loop-run1 --model sonnet

Оснастка вне бота (`src/` её не импортирует). Трекер — обычный Markdown-файл со списком
`- [ ] ID заголовок` (формат — `tasks/tracker.example.md`), который ведёт пользователь. Раннер на
каждой итерации запускает НОВУЮ сессию `claude -p` с одним и тем же промптом
(`exec_loop_lib.PROMPT_TEMPLATE`): ассистент сам открывает трекер, берёт первую невыполненную
задачу, выбирает профиль, выполняет, отмечает `[x]` и коммитит. Человек между задачами ничего не
подсказывает. Раннер только запускает сессии, проверяет результат и ведёт лог.

Приёмка (раннер делает её сам, не доверяя словам ассистента):
1. в трекере появилась хотя бы одна новая отметка `[x]` у задачи, которая была открыта;
2. выполнены команды из строк «Проверка: `cmd`» этих задач (`bash -c`, из корня репозитория;
   в окружении `TASK_BASE` — коммит перед итерацией);
3. проверки проекта (`--gate`; по умолчанию сборка и тесты) не стали хуже: проверка, которая была
   красной ДО итерации, не учитывается.
Не прошло — изменения откатываются, вторая попытка (`--attempts`) видит только вывод приёмки.
Если все попытки провалены, задача помечается `[!]` (раннер коммитит отметку), и цикл идёт дальше.

Итог — `reports/exec_loop/<ветка>/` (`log.md`, `results.json`, вывод каждой попытки; `reports/`
не коммитится). Чистые правила — `exec_loop_lib.py`, тесты — `tests/test_exec_loop_lib.py`.

Ограничения: ассистент работает в ЭТОМ репозитории с правами из `ALLOWED_TOOLS` (+ `--allow`), без
`push`; команды «Проверка:» из трекера выполняются раннером с вашими правами — трекер ведёте вы.
Запускайте на отдельной ветке прогона, не на основной.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exec_loop_lib as lib  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKER = "tasks/tracker.md"
PROTECTED_BRANCHES = ("main", "master")
ENV_STUB = {"TELEGRAM_BOT_TOKEN": "x", "DEEPSEEK_API_KEY": "x"}
CHECK_TIMEOUT_SECONDS = 600
IGNORED_PREFIXES = ("reports/", ".venv", "out/")
DEFAULT_GATES = [
    ".venv/bin/python -m compileall -q src",
    ".venv/bin/pytest tests -q",
]

ALLOWED_TOOLS = [
    "Read", "Edit", "Write", "Grep", "Glob", "Task", "Agent",
    "Bash(.venv/bin/pytest:*)", "Bash(.venv/bin/ruff:*)", "Bash(.venv/bin/python:*)",
    "Bash(TELEGRAM_BOT_TOKEN=x DEEPSEEK_API_KEY=x .venv/bin/pytest:*)",
    "Bash(bash tasks/check.sh:*)", "Bash(tasks/check.sh:*)",
    "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)",
    "Bash(git add:*)", "Bash(git commit:*)",
    "Bash(grep:*)", "Bash(ls:*)", "Bash(wc:*)",
]
DENIED_TOOLS = ["Read(./.env)", "Bash(git push:*)"]


def git(*args: str, check: bool = True) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)
    return done.stdout.strip()


def changed_paths(base: str, tracker: str) -> list[str]:
    """Файлы, изменённые относительно base (коммиты, правки, новые), без служебных путей."""
    tracked = git("diff", "--name-only", base).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    paths = tracked + untracked
    return [p for p in paths if p and p != tracker and not p.startswith(IGNORED_PREFIXES)]


def run_process(cmd: list[str], env: dict[str, str], timeout: float) -> tuple[int, str, bool]:
    """Запуск с таймаутом; при превышении убивается вся группа процессов."""
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out, False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        return -9, out or "", True


def run_shell(command: str, base: str) -> tuple[bool, str]:
    """Команда приёмки через bash из корня репозитория; True — код выхода 0."""
    env = {**os.environ, **ENV_STUB, "TASK_BASE": base}
    for name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(name, None)  # httpx не понимает схему socks:// из окружения разработчика
        env.pop(name.lower(), None)
    code, out, timed_out = run_process(["bash", "-c", command], env, CHECK_TIMEOUT_SECONDS)
    return code == 0 and not timed_out, out


def read_tasks(tracker: Path) -> list[lib.Task]:
    return lib.parse_tracker(tracker.read_text(encoding="utf-8"))


def judge(before: list[lib.Task], tracker: Path, rel: str, base: str, gates_ok: dict[str, bool],
          gates: list[str]) -> tuple[list[lib.Task], bool, str]:
    """Приёмка попытки → (отмеченные задачи, прошла ли, вывод для лога и повтора)."""
    try:
        done = lib.newly_done(before, read_tasks(tracker))
    except ValueError as error:
        return [], False, f"Трекер {rel} не разбирается: {error}"
    if not done:
        return [], False, f"В {rel} ни одна открытая задача не отмечена как выполненная («- [x]»)."
    report, passed = [], True
    for task in done:
        for command in task.checks:
            ok, out = run_shell(command, base)
            verdict = "ок" if ok else "ПРОВАЛ"
            report.append(f"[{task.task_id}] проверка `{command}`: {verdict}\n{out[-1500:]}")
            passed &= ok
    for gate in gates:
        if not gates_ok.get(gate):
            report.append(f"проверка проекта `{gate}` была красной до итерации — пропущена")
            continue
        ok, out = run_shell(gate, base)
        verdict = "ок" if ok else "ПРОВАЛ"
        report.append(f"проверка проекта `{gate}`: {verdict}\n{out[-1500:]}")
        passed &= ok
    return done, passed, "\n".join(report)


def run_attempt(number: int, previous: str | None, args, out_dir: Path, tag: str) -> lib.Attempt:
    base = git("rev-parse", "HEAD")
    cmd = [
        "claude", "-p", lib.build_prompt(args.tracker, previous), "--output-format", "json",
        "--permission-mode", "acceptEdits", "--no-session-persistence",
        "--allowedTools", *ALLOWED_TOOLS, *args.allow,
        "--disallowedTools", *DENIED_TOOLS, *args.deny,
    ]
    if args.model:
        cmd += ["--model", args.model]
    env = {**os.environ, "TASK_BASE": base}
    env.update(dict(item.split("=", 1) for item in args.env))
    started = time.monotonic()
    code, out, timed_out = run_process(cmd, env, args.timeout)
    seconds = time.monotonic() - started
    (out_dir / f"attempt_{tag}_{number}.json").write_text(out, encoding="utf-8")
    data = lib.parse_claude_output(out)
    return lib.Attempt(
        seconds=seconds, returncode=code, timed_out=timed_out, turns=data.get("num_turns"),
        cost_usd=data.get("total_cost_usd"), changed_files=len(changed_paths(base, args.tracker)),
    )


def finish_commit(label: str) -> bool:
    """Приёмка пройдена, но ассистент закоммитил не всё — коммитит раннер. True — было."""
    if not git("status", "--porcelain"):
        return False
    git("add", "-A", "--", ".", ":!reports")
    git("commit", "-q", "-m", f"{label} (commit by exec_loop runner)")
    return True


def rollback(base: str) -> None:
    git("reset", "--hard", "-q", base)
    git("clean", "-fdq", "-e", ".venv", "-e", "reports")


def run_iteration(target: lib.Task, args, tracker: Path, gates: list[str],
                  out_dir: Path) -> lib.TaskResult:
    before = read_tasks(tracker)
    base = git("rev-parse", "HEAD")
    gates_ok = {gate: run_shell(gate, base)[0] for gate in gates}
    result = lib.TaskResult(target.task_id)
    previous = None
    for number in range(1, args.attempts + 1):
        attempt = run_attempt(number, previous, args, out_dir, target.task_id)
        done, passed, report = judge(before, tracker, args.tracker, base, gates_ok, gates)
        attempt.check_passed, attempt.check_output = passed, report
        (out_dir / f"check_{target.task_id}_{number}.txt").write_text(report, encoding="utf-8")
        result.attempts.append(attempt)
        if passed:
            ids = [t.task_id for t in done]
            result.task_id = ",".join(ids)
            if ids != [target.task_id]:
                result.note = f"ожидалась {target.task_id}, отмечено: {', '.join(ids)}"
            result.passed = True
            result.commit_by_runner = finish_commit(f"{result.task_id}: {done[0].title}")
            if git("rev-parse", "HEAD") == base:
                result.passed, result.note = False, "приёмка пройдена, но коммита нет"
            return result
        previous = report
        rollback(base)
    blocked = lib.mark_blocked(tracker.read_text(encoding="utf-8"), target)
    tracker.write_text(blocked, encoding="utf-8")
    git("add", args.tracker)
    note = f"exec_loop: {target.task_id} заблокирована после {args.attempts} попыток"
    git("commit", "-q", "-m", note)
    return result


def write_report(out_dir: Path, name: str, model: str, results: list[lib.TaskResult]) -> None:
    (out_dir / "log.md").write_text(lib.render_log(name, model, results), encoding="utf-8")
    dump = [{**asdict(r), "passed": r.passed, "failure_class": r.failure_class} for r in results]
    text = json.dumps(dump, ensure_ascii=False, indent=1)
    (out_dir / "results.json").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help=f"трекер ({DEFAULT_TRACKER})")
    parser.add_argument("--branch", help="создать ветку прогона и перейти в неё")
    parser.add_argument("--from", dest="start", default="HEAD", help="откуда создать ветку (HEAD)")
    parser.add_argument("--model", help="модель для claude --model")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE для процесса claude")
    parser.add_argument("--timeout", type=float, default=900, help="секунд на попытку")
    parser.add_argument("--attempts", type=int, default=2, help="попыток на задачу")
    parser.add_argument("--max-tasks", type=int, default=0, help="остановиться после N итераций")
    parser.add_argument("--stop-on-fail", action="store_true", help="стоп после первого провала")
    parser.add_argument("--gate", action="append", default=None,
                        help="проверка проекта (повторяемая; по умолчанию сборка и тесты)")
    parser.add_argument("--no-gates", action="store_true", help="не запускать проверки проекта")
    parser.add_argument("--allow", action="append", default=[], help="доп. разрешённый инструмент")
    parser.add_argument("--deny", action="append", default=[], help="доп. запрещённый инструмент")
    parser.add_argument("--dry-run", action="store_true", help="показать задачи и промпт")
    args = parser.parse_args()

    tracker = ROOT / args.tracker
    tasks = read_tasks(tracker)
    gates = [] if args.no_gates else (args.gate if args.gate is not None else DEFAULT_GATES)
    if args.dry_run:
        for task in tasks:
            checks = "; ".join(task.checks) or "—"
            print(f"[{task.status:7}] {task.task_id}: {task.title}  (проверка: {checks})")
        print(f"\n===== промпт\n{lib.build_prompt(args.tracker)}")
        return 0
    if git("status", "--porcelain"):
        sys.exit("Рабочая копия не чистая: закоммитьте или уберите изменения (трекер тоже).")
    if args.branch:
        git("switch", "-c", args.branch, args.start)
    branch = git("branch", "--show-current")
    if not branch or branch in PROTECTED_BRANCHES:
        sys.exit(f"Запуск на ветке {branch!r} запрещён: используйте ветку прогона (--branch).")

    out_dir = ROOT / "reports" / "exec_loop" / branch.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[lib.TaskResult] = []
    while True:
        target = lib.first_open(read_tasks(tracker))
        if target is None or (args.max_tasks and len(results) >= args.max_tasks):
            break
        print(f"[{target.task_id}] {target.title} …", flush=True)
        result = run_iteration(target, args, tracker, gates, out_dir)
        results.append(result)
        write_report(out_dir, branch, args.model or "по умолчанию", results)
        verdict = "готово" if result.passed else "ПРОВАЛ"
        print(f"[{result.task_id}] {verdict} за {result.seconds:.0f} с", flush=True)
        if args.stop_on_fail and not result.passed:
            break
    metrics = lib.compute_metrics(results)
    print(f"Подряд без паузы: {metrics.streak} ({metrics.streak_minutes:.1f} мин); "
          f"выполнено {metrics.passed}/{metrics.total}")
    print(f"Отчёт: {out_dir}/log.md" if results else "Невыполненных задач в трекере нет.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
