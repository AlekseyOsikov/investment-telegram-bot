"""Execution loop (День 5): пул задач `tasks/backlog.md` → ассистент → приёмка → коммит → следующая.

    python scripts/exec_loop.py --branch exec-loop-run1 --model sonnet
    python scripts/exec_loop.py --branch exec-loop-local1 --model qwen3-coder:30b \\
        --env ANTHROPIC_BASE_URL=http://localhost:11434 --env ANTHROPIC_AUTH_TOKEN=ollama \\
        --timeout 1800

Оснастка вне бота (`src/` её не импортирует). На каждую задачу — отдельный `claude -p` в СВЕЖЕЙ
сессии с одним и тем же шаблоном промпта (`exec_loop_lib.PROMPT_TEMPLATE`): между задачами человек
ничего не подсказывает, профиль выбирает ассистент. Приёмку делает `tasks/check.sh` (раннер
запускает её сам, не доверяя словам ассистента). Не прошло — изменения откатываются и идёт вторая
попытка, в которой ассистент видит ТОЛЬКО вывод приёмки. Итог — `reports/exec_loop/<ветка>/`
(`log.md`, `results.json`, вывод каждой попытки; `reports/` не коммитится). Чистые правила —
`exec_loop_lib.py`, тесты — `tests/test_exec_loop_lib.py`.

Ограничения: ассистент работает в ЭТОМ репозитории с правами из `ALLOWED_TOOLS` (без `push`,
без произвольных команд); запускайте только на ветке прогона. Запрет читать `tasks/check.sh`
держится на `--disallowedTools` и промпте, обход через оболочку не исключён.
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
BACKLOG = ROOT / "tasks" / "backlog.md"
POOL_TAG = "exec-loop-pool"
ENV_STUB = {"TELEGRAM_BOT_TOKEN": "x", "DEEPSEEK_API_KEY": "x"}
CHECK_TIMEOUT_SECONDS = 600
IGNORED_PREFIXES = ("tasks/", "reports/", ".venv", "out/")

ALLOWED_TOOLS = [
    "Read", "Edit", "Write", "Grep", "Glob", "Task", "Agent",
    "Bash(.venv/bin/pytest:*)", "Bash(.venv/bin/ruff:*)", "Bash(.venv/bin/python:*)",
    "Bash(TELEGRAM_BOT_TOKEN=x DEEPSEEK_API_KEY=x .venv/bin/pytest:*)",
    "Bash(tasks/check.sh:*)", "Bash(bash tasks/check.sh:*)",
    "Bash(git status:*)", "Bash(git diff:*)", "Bash(git log:*)",
    "Bash(git add:*)", "Bash(git commit:*)",
    "Bash(grep:*)", "Bash(ls:*)", "Bash(wc:*)",
]
DENIED_TOOLS = ["Read(./tasks/check.sh)", "Read(./.env)", "Read(./.env.*)", "Bash(git push:*)"]


def git(*args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=check
    )
    return done.stdout.strip()


def changed_paths(base: str) -> list[str]:
    """Файлы, изменённые относительно base (коммиты, правки, новые), без служебных путей."""
    tracked = git("diff", "--name-only", base).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return [p for p in tracked + untracked if p and not p.startswith(IGNORED_PREFIXES)]


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


def run_check(task_id: str, base: str) -> tuple[bool, str]:
    env = {**os.environ, **ENV_STUB, "TASK_BASE": base}
    cmd = ["bash", "tasks/check.sh", task_id]
    code, out, timed_out = run_process(cmd, env, CHECK_TIMEOUT_SECONDS)
    return code == 0 and not timed_out, out


def run_attempt(
    task: lib.Task, number: int, previous: str | None, args, out_dir: Path
) -> lib.Attempt:
    base = git("rev-parse", "HEAD")
    cmd = [
        "claude", "-p", lib.build_prompt(task, previous), "--output-format", "json",
        "--permission-mode", "acceptEdits", "--no-session-persistence",
        "--allowedTools", *ALLOWED_TOOLS, "--disallowedTools", *DENIED_TOOLS,
    ]
    if args.model:
        cmd += ["--model", args.model]
    env = {**os.environ, "TASK_BASE": base}
    env.update(dict(item.split("=", 1) for item in args.env))
    started = time.monotonic()
    code, out, timed_out = run_process(cmd, env, args.timeout)
    seconds = time.monotonic() - started
    (out_dir / f"attempt_{task.task_id}_{number}.json").write_text(out, encoding="utf-8")
    data = lib.parse_claude_output(out)

    passed, check_out = run_check(task.task_id, base)
    (out_dir / f"check_{task.task_id}_{number}.txt").write_text(check_out, encoding="utf-8")
    return lib.Attempt(
        seconds=seconds, returncode=code, timed_out=timed_out, turns=data.get("num_turns"),
        cost_usd=data.get("total_cost_usd"), changed_files=len(changed_paths(base)),
        check_passed=passed, check_output=check_out,
    )


def finish_commit(task: lib.Task) -> bool:
    """Если приёмка пройдена, но ассистент не закоммитил всё сам — коммитит раннер. True — было."""
    if not git("status", "--porcelain"):
        return False
    git("add", "-A", "--", ".", ":!tasks", ":!reports")
    git("commit", "-q", "-m", f"{task.task_id}: {task.title} (commit by exec_loop runner)")
    return True


def rollback(base: str) -> None:
    git("reset", "--hard", "-q", base)
    git("clean", "-fdq", "-e", ".venv", "-e", "reports")


def run_task(task: lib.Task, args, out_dir: Path) -> lib.TaskResult:
    result = lib.TaskResult(task.task_id)
    base = git("rev-parse", "HEAD")
    previous = None
    for number in range(1, args.attempts + 1):
        attempt = run_attempt(task, number, previous, args, out_dir)
        result.attempts.append(attempt)
        if attempt.check_passed:
            result.passed = True
            result.commit_by_runner = finish_commit(task)
            if git("rev-parse", "HEAD") == base:  # приёмка пройдена без изменений — не засчитываем
                result.passed = False
            return result
        previous = attempt.check_output
        rollback(base)
    return result


def write_report(out_dir: Path, args, results: list[lib.TaskResult]) -> None:
    name = args.branch or git("branch", "--show-current")
    model = args.model or "по умолчанию"
    (out_dir / "log.md").write_text(lib.render_log(name, model, results), encoding="utf-8")
    dump = [{**asdict(r), "passed": r.passed, "failure_class": r.failure_class} for r in results]
    text = json.dumps(dump, ensure_ascii=False, indent=1)
    (out_dir / "results.json").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--branch", help="создать ветку прогона от тега из --from")
    parser.add_argument("--from", dest="pool_tag", default=POOL_TAG, help=f"тег пула ({POOL_TAG})")
    parser.add_argument("--tasks", help="T03, T03-T07, T01,T05 (по умолчанию все)")
    parser.add_argument("--model", help="модель для claude --model")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE для процесса claude")
    parser.add_argument("--timeout", type=float, default=900, help="секунд на попытку")
    parser.add_argument("--attempts", type=int, default=2, help="попыток на задачу")
    parser.add_argument("--stop-on-fail", action="store_true", help="стоп после первого провала")
    parser.add_argument("--dry-run", action="store_true", help="только показать промпты")
    args = parser.parse_args()

    tasks = lib.select_tasks(lib.parse_backlog(BACKLOG.read_text(encoding="utf-8")), args.tasks)
    if args.dry_run:
        for task in tasks:
            print(f"===== {task.task_id}\n{lib.build_prompt(task)}")
        return 0
    if git("status", "--porcelain"):
        sys.exit("Рабочая копия не чистая: закоммитьте или уберите изменения.")
    if args.branch:
        git("switch", "-c", args.branch, args.pool_tag)
    branch = git("branch", "--show-current")
    if not branch.startswith("exec-loop-") or branch in ("exec-loop-base",):
        sys.exit(f"Запуск только на ветке прогона exec-loop-*, сейчас: {branch!r} (см. --branch).")

    out_dir = ROOT / "reports" / "exec_loop" / branch
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[lib.TaskResult] = []
    for task in tasks:
        print(f"[{task.task_id}] {task.title} …", flush=True)
        result = run_task(task, args, out_dir)
        results.append(result)
        write_report(out_dir, args, results)
        verdict = "готово" if result.passed else "ПРОВАЛ"
        print(f"[{task.task_id}] {verdict} за {result.seconds:.0f} с", flush=True)
        if args.stop_on_fail and not result.passed:
            break
    metrics = lib.compute_metrics(results)
    print(f"Подряд без паузы: {metrics.streak}; выполнено {metrics.passed}/{metrics.total}")
    print(f"Отчёт: {out_dir}/log.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
