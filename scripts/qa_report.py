"""QA-отчёт по PR: собирает итоги `scripts/qa_ci.sh` в один комментарий.

    python scripts/qa_report.py --dir qa-out --sha <sha> [--pr 42] [--run-url <url>] [--dry-run]

Оснастка вне бота (`src/` её не импортирует). Читает каталог результатов, пишет `report.md` рядом с
ними, добавляет отчёт в Job summary (`GITHUB_STEP_SUMMARY`) и, если задан `--pr`, публикует ОДИН
обновляемый комментарий через `gh api` (`GH_TOKEN`, `GITHUB_REPOSITORY` задаёт GitHub Actions).
У PR из форка токен только для чтения: публикация не удаётся, и это не ошибка — отчёт остаётся в
Job summary и в артефакте. Чистые правила — `qa_report_lib.py`, тесты —
`tests/test_qa_report_lib.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qa_report_lib as lib  # noqa: E402

BOT_LOGIN = "github-actions[bot]"
GH_TIMEOUT_SECONDS = 60


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def load_input(directory: Path, sha: str, run_url: str | None) -> lib.QaInput:
    changed_text = read_text(directory / "changed.txt") or ""
    return lib.QaInput(
        sha=sha,
        status=lib.parse_status(read_text(directory / "status.txt") or ""),
        pytest_text=read_text(directory / "pytest.txt") or "",
        smoke_report=read_text(directory / "smoke" / "report.md"),
        changed=tuple(line for line in changed_text.splitlines() if line.strip()),
        run_url=run_url,
    )


def run_gh(args: list[str], *, stdin: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["gh", *args],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=GH_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("не найден `gh`") from None
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:2])} завершился с кодом {result.returncode}")
    return result.stdout


def find_comment(repo: str, number: int) -> int | None:
    """id комментария QA: тело начинается с маркера, автор — бот (чужой текст с таким началом не
    перезаписывается)."""
    jq = (
        f'.[] | select(.user.login == "{BOT_LOGIN}" and (.body | startswith("{lib.MARKER}")))'
        " | .id"
    )
    out = run_gh(["api", "--paginate", f"repos/{repo}/issues/{number}/comments", "--jq", jq])
    ids = [line.strip() for line in out.splitlines() if line.strip().isdigit()]
    return int(ids[0]) if ids else None


def publish_comment(repo: str, number: int, body: str) -> str:
    payload = json.dumps({"body": body}, ensure_ascii=False)
    existing = find_comment(repo, number)
    if existing is not None:
        run_gh(
            ["api", "--method", "PATCH", f"repos/{repo}/issues/comments/{existing}",
             "--input", "-"],
            stdin=payload,
        )  # fmt: skip
        return "обновлён"
    run_gh(
        ["api", "--method", "POST", f"repos/{repo}/issues/{number}/comments", "--input", "-"],
        stdin=payload,
    )
    return "создан"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True, help="каталог результатов qa_ci.sh")
    parser.add_argument("--sha", default="", help="коммит, который проверялся")
    parser.add_argument("--pr", type=int, default=0, help="номер PR (0 — не публиковать)")
    parser.add_argument("--run-url", default="", help="ссылка на прогон Actions")
    parser.add_argument("--dry-run", action="store_true", help="печать вместо публикации")
    args = parser.parse_args()

    directory = Path(args.dir)
    data = load_input(directory, args.sha, args.run_url or None)
    body = lib.build_comment(data)
    (directory / "report.md").write_text(body, encoding="utf-8")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path and not args.dry_run:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(body.removeprefix(lib.MARKER).lstrip("\n"))

    if args.dry_run or not args.pr:
        print(body)
        return 0

    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not repo:
        print("GITHUB_REPOSITORY не задан: комментарий не опубликован.")
        return 0
    try:
        action = publish_comment(repo, args.pr, body)
    except RuntimeError as exc:
        # Токен PR из форка только для чтения — это штатная ситуация, а не сбой проверок.
        print(f"Комментарий не опубликован ({exc}); отчёт — в Job summary и артефакте.")
        return 0
    print(f"Комментарий QA {action}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
