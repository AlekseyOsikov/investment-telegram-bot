"""День 1: однопроходная генерация команды `/terms` локальной моделью (не часть бота).

Модель не видит репозиторий, поэтому ей подаётся то, что пользователь IDE приложил бы сам:
правила (системное сообщение), эталонные выдержки и якорные фрагменты `main.py`/`README.md`/
`CLAUDE.md`. Ответ — новые файлы целиком и блоки SEARCH/REPLACE; скрипт применяет их в
ЧИСТОМ worktree и запускает проверки (`compileall`, `ruff`, `pytest`, чек-лист структуры).
Промпт — `scripts/ollama/prompts/terms.txt` (дословно из эксперимента «Rules»), стартовый
коммит — `a8cff69`. Отчёт — `docs/local-llm.md`.

Запуск (worktree должен быть чистым, с `.venv`):
    git worktree add --detach ../bot-terms a8cff69 && ln -s "$PWD/.venv" ../bot-terms/.venv
    python scripts/ollama/terms_oneshot.py ../bot-terms out/terms rules qwen3-coder:30b
    python scripts/ollama/terms_oneshot.py ../bot-terms2 out/terms2 norules qwen3-coder:30b
    python scripts/ollama/terms_oneshot.py ../bot-terms out/terms rules qwen3-coder:30b --reuse
(`--reuse` повторно применяет сохранённый `answer.txt` без обращения к модели.)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import local_llm_client as client
import local_llm_lib as lib

ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = ROOT / "scripts" / "ollama" / "prompts" / "terms.txt"
RULES_FILE = ROOT / ".continue" / "rules" / "code-style.md"
NEUTRAL_SYSTEM = "Ты помощник-программист. Отвечай по-русски."
TEST_ENV = {"PATH": "/usr/bin:/bin", "TELEGRAM_BOT_TOKEN": "x", "DEEPSEEK_API_KEY": "x"}

FORMAT = """## Формат ответа (строго)
Новый файл целиком:
=== FILE: путь/к/файлу ===
<содержимое>
=== END ===

Правка существующего файла (SEARCH — дословный фрагмент из приведённых выше, один раз):
=== EDIT: путь/к/файлу ===
<<<<<<< SEARCH
<старый текст>
=======
<новый текст>
>>>>>>> REPLACE
=== END ===
Других пояснений не нужно."""


def around(root: Path, rel: str, needle: str, before: int = 2, after: int = 2) -> str:
    """Строки файла вокруг первой строки с `needle`."""
    lines = (root / rel).read_text(encoding="utf-8").splitlines()
    i = next(n for n, line in enumerate(lines) if needle in line)
    return "\n".join(lines[max(0, i - before) : i + after + 1])


def head(root: Path, rel: str, count: int) -> str:
    return "\n".join((root / rel).read_text(encoding="utf-8").splitlines()[:count])


def context(root: Path) -> str:
    """Эталоны и якорные фрагменты (SEARCH модель копирует из них)."""
    queue = "src/llm_queue/commands.py"
    return f"""## Эталоны (реальный код проекта)

### src/news_digest/rules.py (начало)
```python
{head(root, "src/news_digest/rules.py", 32)}
```

### src/news_digest/texts.py (начало)
```python
{head(root, "src/news_digest/texts.py", 32)}
```

### {queue} (фабрика обработчиков)
```python
{around(root, queue, "def build_queue_handlers", 0, 12)}
```

### tests/test_news_digest_texts.py (тест запретных формулировок)
```python
{around(root, "tests/test_news_digest_texts.py", "GUARANTEE", 1, 20)}
```

## Фрагменты существующих файлов, которые нужно править (SEARCH копируй дословно)

### src/main.py — импорты
```
{around(root, "src/main.py", "from support_assistant.command import build_support_handlers", 2, 1)}
```
### src/main.py — справка /help
```
{around(root, "src/main.py", '"/queue_cancel [номер]', 1, 3)}
```
### src/main.py — регистрация
```
{around(root, "src/main.py", "for handler in build_queue_handlers():", 1, 3)}
```
### README.md — дерево
```
{around(root, "README.md", "# /queue, /queue_cancel, постановка задач", 2, 1)}
```
### README.md — таблица команд
```
{around(root, "README.md", "| `/queue_cancel [номер]`", 1, 1)}
```
### CLAUDE.md — список вложенных CLAUDE.md
```
{around(root, "CLAUDE.md", "`src/research/`, `src/support_assistant/`.", 1, 1)}
```
### CLAUDE.md — карта src/
```
{around(root, "CLAUDE.md", "`/queue_cancel` (`commands.py`)", 1, 1)}
```
"""


def generate(args: argparse.Namespace, system: str, user: str) -> tuple[str, dict]:
    """Запрос к модели; возвращает ответ и метрики генерации."""
    body = {
        "model": args.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"num_ctx": args.num_ctx, "temperature": 0.2, "top_p": 0.9, "num_predict": 8192},
    }
    think = lib.default_think(args.model)
    if think is not None:
        body["think"] = think
    started = time.time()
    data = client.chat(args.url, body)
    metrics = {
        "model": args.model,
        "variant": args.variant,
        "num_ctx": args.num_ctx,
        "wall_s": round(time.time() - started, 1),
        "prompt_tokens": data["prompt_eval_count"],
        "gen_tokens": data["eval_count"],
        "prefill_s": round(data["prompt_eval_duration"] / 1e9, 1),
        "gen_tps": lib.rate(data["eval_count"], data["eval_duration"]),
        "done_reason": data.get("done_reason"),  # `length` — ответ оборван лимитом токенов
    }
    return data["message"]["content"], metrics


def run(root: Path, cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, cwd=root, env=TEST_ENV, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def checks(root: Path) -> dict:
    """Проверки результата: компиляция, ruff, pytest, запуск фабрики, чек-лист структуры."""
    status = run(root, ["git", "status", "--porcelain", "--untracked-files=all"])[1]
    changed = [line[3:] for line in status.splitlines() if ".venv" not in line]
    py = [f for f in changed if f.endswith(".py")]
    venv = root / ".venv" / "bin"
    ruff_cmd = [str(venv / "ruff"), "check", *py, "--output-format", "concise"]
    ruff = run(root, ruff_cmd)[1] if py else ""
    pytest = run(root, [str(venv / "pytest"), "tests/", "-q", "-p", "no:cacheprovider"])[1]
    src = {f: (root / f).read_text(encoding="utf-8") for f in changed if f.startswith("src/terms/")}
    src = {f: t for f, t in src.items() if f.endswith(".py")}
    main_py = (root / "src/main.py").read_text(encoding="utf-8")
    code = "\n".join(src.values())
    return {
        "changed_files": changed,
        "compileall_rc": run(root, [str(venv / "python"), "-m", "compileall", "-q", "src"])[0],
        "ruff_out": ruff[-1500:],
        "pytest_tail": "\n".join(pytest.strip().splitlines()[-4:]),
        "checklist": {
            "rules/texts/commands": all(
                f"src/terms/{n}.py" in src for n in ("rules", "texts", "commands")),
            "handler_factory": "def build_terms_handler" in code,
            "registered_in_main": "build_terms_handler" in main_py,
            "help_line": "/terms" in main_py.split("def help_command")[1][:6000],
            "readme_row": "/terms" in (root / "README.md").read_text(encoding="utf-8"),
            "nested_claude_md": "src/terms/CLAUDE.md" in changed,
            "private_filter": "ChatType.PRIVATE" in code,
            "future_annotations": bool(src) and all(
                "from __future__ import annotations" in t for t in src.values() if t.strip()),
            "disclaimer": "лицензированн" in code,
            "tests_exist": any(f.startswith("tests/") for f in changed),
            "forbidden_words_test": any(
                re.search(r"гарантир|GUARANTEE", (root / f).read_text(encoding="utf-8"))
                for f in changed
                if f.startswith("tests/")
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("worktree", type=Path, help="чистый worktree на a8cff69 с .venv")
    parser.add_argument("out", type=Path, help="каталог для request.json, answer.txt, metrics.json")
    parser.add_argument("variant", choices=["rules", "norules"])
    parser.add_argument("model")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--url", default=client.DEFAULT_URL)
    parser.add_argument("--reuse", action="store_true", help="применить сохранённый answer.txt")
    args = parser.parse_args()

    root, out = args.worktree.resolve(), args.out
    out.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    rules = lib.profile_body(RULES_FILE.read_text(encoding="utf-8"))
    system = rules if args.variant == "rules" else NEUTRAL_SYSTEM
    user = f"{prompt}\n\n{FORMAT}\n\n{context(root)}"
    (out / "request.json").write_text(
        json.dumps({"system": system, "user": user}, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if args.reuse:
        answer = (out / "answer.txt").read_text(encoding="utf-8")
        old = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        metrics = {k: old[k] for k in ("model", "variant", "num_ctx", "wall_s", "prompt_tokens",
                                       "gen_tokens", "prefill_s", "gen_tps", "done_reason")}
    else:
        answer, metrics = generate(args, system, user)
        (out / "answer.txt").write_text(answer, encoding="utf-8")

    def read(rel: str) -> str | None:
        path = root / rel
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def write(rel: str, text: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    metrics["apply_problems"] = lib.apply_answer(answer, read, write)
    metrics.update(checks(root))
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
