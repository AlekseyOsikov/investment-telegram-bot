"""Независимая оценка прогона bug-fix (`agent_loop.py bugfix`): что реально сделано в worktree,
а не что модель написала в отчёте (не часть бота). Одна JSON-строка в stdout.

Проверяется: исправлено ли поведение на трёх входах (`parse_watch_args`), падает ли новый тест на
СТАРОМ коде (правка `src/` временно откатывается и возвращается), зелёный ли весь `pytest`,
не вырос ли `ruff` по изменённым файлам, не удалялись ли строки существующих тестов, затронуты
ли файлы вне `src/` и `tests/`, а также процесс по журналу инструментов: воспроизведён ли баг и
запускались ли тесты ДО первой успешной правки, сколько неудачных `edit_file`, все ли секции отчёта.

Запуск (worktree после прогона и каталог его результатов):
    python scripts/ollama/bugfix_eval.py ../bot-bug out/bug
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ENV = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TELEGRAM_BOT_TOKEN": "x",
       "DEEPSEEK_API_KEY": "x", "PYTHONPATH": "src"}
EXPECTED_BEHAVIOR = "['ParseError', 'ParseError', 'WatchRequest']"
SECTIONS = ("Симптом", "Причина", "Что починил", "Что проверил", "Что НЕ проверено")
BEHAVIOR_CODE = (
    "from price_watch.parse import parse_watch_args as p\n"
    "print([type(p(a)).__name__ for a in ("
    "['SBER','١٥m','1h'], ['SBER','15m','١h'], ['SBER','15m','1h'])])"
)


def sh(root: Path, cmd: list[str], timeout: int = 300) -> tuple[int, str]:
    done = subprocess.run(cmd, cwd=root, env=ENV, capture_output=True, text=True, timeout=timeout)
    return done.returncode, done.stdout + done.stderr


def ruff_count(root: Path, paths: list[str]) -> int:
    if not paths:
        return 0
    out = sh(root, [str(root / ".venv/bin/ruff"), "check", "--output-format", "concise", *paths])[1]
    found = re.search(r"Found (\d+) error", out)
    return int(found.group(1)) if found else 0


def new_test_fails_on_old_code(root: Path, src_patch: str) -> bool:
    """Откатить правку `src/`, прогнать тесты файла, вернуть правку."""
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as handle:
        handle.write(src_patch)
    try:
        sh(root, ["git", "apply", "-R", handle.name])
        tail = sh(root, [str(root / ".venv/bin/pytest"), "tests/test_price_watch_parse.py", "-q",
                         "-p", "no:cacheprovider"])[1].strip().splitlines()
    finally:
        sh(root, ["git", "apply", handle.name])
        Path(handle.name).unlink()
    return bool(tail) and " failed" in tail[-1]


def process_from_log(calls: list[dict]) -> dict:
    """Порядок шагов по журналу инструментов."""
    tools = [c["tool"] for c in calls]
    first_edit = next(
        (i for i, c in enumerate(calls)
         if c["tool"] in ("edit_file", "append_file") and c["result_head"] == "ok"), None)
    before = tools[:first_edit] if first_edit is not None else tools
    return {
        "reproduced_before_edit": "run_python" in before,
        "tests_run_before_fix": "run_pytest" in before,
        "edit_errors": sum(c["tool"] == "edit_file" and c["result_head"].startswith("ошибка")
                           for c in calls),
    }


def evaluate(root: Path, out: Path) -> dict:
    python = str(root / ".venv/bin/python")
    changed = [line[3:] for line in sh(root, ["git", "status", "--porcelain"])[1].splitlines()
               if ".venv" not in line]
    result: dict = {"run": out.name, "changed_files": changed}
    result["only_src_tests"] = (all(f.startswith(("src/", "tests/")) for f in changed)
                                if changed else None)
    result["behavior"] = sh(root, [python, "-c", BEHAVIOR_CODE])[1].strip().splitlines()[-1]
    result["fixed"] = result["behavior"] == EXPECTED_BEHAVIOR
    full = sh(root, [str(root / ".venv/bin/pytest"), "tests/", "-q", "-p", "no:cacheprovider"])
    result["pytest_full"] = (full[1].strip().splitlines() or [""])[-1]

    src_patch = sh(root, ["git", "diff", "--", "src"])[1]
    tests_diff = sh(root, ["git", "diff", "--", "tests"])[1]
    lines = tests_diff.splitlines()
    result["tests_deleted_lines"] = sum(
        x.startswith("-") and not x.startswith("---") for x in lines)
    result["tests_added_lines"] = sum(x.startswith("+") and not x.startswith("+++") for x in lines)
    result["new_test_fails_on_old_code"] = (
        new_test_fails_on_old_code(root, src_patch) if src_patch.strip() and tests_diff.strip()
        else None)

    py = [f for f in changed if f.endswith(".py")]
    with tempfile.TemporaryDirectory() as tmp:
        olds = []
        for f in py:
            code, text = sh(root, ["git", "show", f"HEAD:{f}"])
            if code == 0:
                path = Path(tmp) / f
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                olds.append(str(path))
        result["ruff_before"], result["ruff_after"] = ruff_count(root, olds), ruff_count(root, py)

    calls = json.loads((out / "toolcalls.json").read_text(encoding="utf-8"))
    result.update(process_from_log(calls))
    answer = (out / "answer.md").read_text(encoding="utf-8")
    result["answer_sections"] = sum(s in answer for s in SECTIONS)
    result["mentions_texts_py_sibling"] = "texts.py" in answer
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    result.update({"turns": metrics["turns"], "wall_s": metrics["wall_s"],
                   "stop": metrics["stop_reason"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("worktree", type=Path)
    parser.add_argument("out", type=Path, help="каталог результатов agent_loop.py")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.worktree.resolve(), args.out), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
