"""День 2: профили research и bug-fix на локальной модели — цикл с инструментами (не часть бота).

Модель вызывает инструменты через нативный tool calling Ollama, скрипт их выполняет. Системное
сообщение — тело профиля: `--variant v1` — облачный `.claude/agents/<профиль>.md` без изменений,
`--variant v2` — `.continue/rules/<профиль>-profile.md` (то же + правила для локальной модели).
Строка про инструменты добавляется к обоим. Отчёт — `docs/local-llm.md`.

Границы (скрипт даёт модели только это):
- research: `list_dir`, `read_file`, `grep` — только чтение, записи нет вовсе;
- bugfix: то же + `edit_file`/`append_file` ТОЛЬКО в `src/` и `tests/` корня, `run_python`
  (`python -c`, PYTHONPATH=src), `run_pytest`, `run_ruff`, `run_compileall`, `git_diff`.
  Оболочки нет, команды запускаются списком аргументов с очищенным окружением и таймаутом;
  `.env*`, `data/`, `.git/`, `.venv/` закрыты. Тесты и `python -c` — код модели, поэтому
  bug-fix запускайте в ОДНОРАЗОВОМ worktree, не в рабочей копии.

Запуск из корня проекта (нужен запущенный Ollama):
    python scripts/ollama/agent_loop.py research . out/research gpt-oss:20b --variant v2
    git worktree add --detach ../bot-bug 763b1e2 && ln -s "$PWD/.venv" ../bot-bug/.venv
    python scripts/ollama/agent_loop.py bugfix ../bot-bug out/bug gpt-oss:20b --variant v2
Баг для bugfix — на `763b1e2` (до исправления `\\d` в `price_watch/parse.py`); оценка результата
независимо от слов модели — `scripts/ollama/bugfix_eval.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import local_llm_client as client
import local_llm_lib as lib

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]  # репозиторий со скриптами и .continue/rules
NUM_CTX = 16384
TRIM_AT = 11000  # старые результаты инструментов заменяются заглушкой
HARD_STOP = 14500  # дальше окно закончится: просим итоговый ответ
MODEL_RETRIES = 2  # повторы запроса при ошибке сервера (например, неразбираемый вызов инструмента)
STUB_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TELEGRAM_BOT_TOKEN": "x",
            "DEEPSEEK_API_KEY": "x", "PYTHONPATH": "src"}
GENERIC_NAMES = {"cli.py", "command.py", "commands.py", "core.py"}
RESEARCH_EXCLUDED = ("research", "agents")

RESEARCH_TASK = (
    "Какие модули в src/ (кроме research/ и agents/) не покрыты тестами, и как устроена очередь "
    "llm_queue: путь вопроса из handle_message до ответа, где применяются лимиты и что "
    "происходит при отмене."
)
BUGFIX_TASK = (
    "Баг: команда /watch принимает периоды с не-ASCII цифрами и пропускает их к серверу. "
    "parse_watch_args(['SBER','١٥m','1h']) возвращает WatchRequest, а не ParseError. "
    "Найди причину, почини, проверь."
)
READ_TOOLS = ["list_dir", "read_file", "grep"]
BUGFIX_TOOLS = [*READ_TOOLS, "edit_file", "append_file", "run_python", "run_pytest", "run_ruff",
                "run_compileall", "git_diff"]


@dataclass(frozen=True)
class Profile:
    agent_file: str
    rule_file: str
    task: str
    tools: list[str]
    writable: tuple[str, ...]
    max_turns: int
    tools_note: str


PROFILES = {
    "research": Profile(
        "research.md", "research-profile.md", RESEARCH_TASK, READ_TOOLS, (), 30,
        "Доступны только `list_dir`, `read_file` (с номерами строк) и `grep` (регулярное "
        "выражение по текстовым файлам). Команд оболочки нет: покрытие определяй через `grep` "
        "по `tests/`. Пути — от корня репозитория.",
    ),
    "bugfix": Profile(
        "bug-fix.md", "bugfix-profile.md", BUGFIX_TASK, BUGFIX_TOOLS, ("src", "tests"), 40,
        "`list_dir`, `read_file` (с номерами строк), `grep`, `edit_file` (точная замена "
        "уникального фрагмента; пути только в src/ и tests/), `append_file` (дописать в конец "
        "файла в src/ или tests/), `run_python` (`python -c`, PYTHONPATH=src), `run_pytest` "
        "(аргументы pytest, токены-заглушки подставлены сами), `run_ruff` (пути), "
        "`run_compileall`, `git_diff`. Оболочки нет. Пути — от корня репозитория.",
    ),
}


class Workspace:
    """Инструменты над одним корнем; доступ определяют `lib.is_denied`/`lib.is_writable`."""

    def __init__(self, root: Path, writable: tuple[str, ...]) -> None:
        self.root = root.resolve()
        self.writable = writable

    def _path(self, rel: str, write: bool = False) -> Path | None:
        parts = lib.split_rel_path(rel)
        if parts is None or lib.is_denied(parts):
            return None
        if write and not lib.is_writable(parts, self.writable):
            return None
        path = (self.root / Path(*parts)).resolve()
        return path if self.root in path.parents or path == self.root else None

    def list_dir(self, path: str = ".") -> str:
        target = self._path(path)
        if not target or not target.is_dir():
            return f"ошибка: каталог недоступен: {path}"
        items = [x for x in sorted(target.iterdir()) if not lib.is_denied((x.name,))]
        return "\n".join(f"{x.name}{'/' if x.is_dir() else ''}" for x in items)[:5000]

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        target = self._path(path)
        if not target or not target.is_file():
            return f"ошибка: файл недоступен: {path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        return lib.read_report(path, text, start_line, end_line)

    @staticmethod
    def _text_files(base: Path):
        """Текстовые файлы под `base`; закрытые каталоги отсекаются до обхода."""
        if base.is_file():
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if not lib.is_denied((d,)))
            for name in sorted(filenames):
                if not lib.is_denied((name,)) and Path(name).suffix in lib.TEXT_SUFFIXES:
                    yield Path(dirpath) / name

    def grep(self, pattern: str, path: str = ".") -> str:
        base = self._path(path)
        if not base:
            return f"ошибка: путь недоступен: {path}"
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"ошибка в регулярном выражении: {exc}"
        hits: list[str] = []
        for f in self._text_files(base):
            hits += lib.grep_text(str(f.relative_to(self.root)),
                                  f.read_text(encoding="utf-8", errors="replace"), rx)
            if len(hits) > lib.MAX_GREP_HITS:
                break
        return lib.grep_report(hits)

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        target = self._path(path, write=True)
        if not target or not target.is_file():
            return f"ошибка: правка запрещена или файл не найден: {path} (пишите в src/, tests/)"
        new, count = lib.replace_once(target.read_text(encoding="utf-8"), old_string, new_string)
        if new is None:
            return f"ошибка: old_string найден {count} раз (нужно ровно 1)"
        target.write_text(new, encoding="utf-8")
        return "ok"

    def append_file(self, path: str, text: str) -> str:
        target = self._path(path, write=True)
        if not target or not target.is_file():
            return f"ошибка: запись запрещена или файл не найден: {path}"
        with target.open("a", encoding="utf-8") as f:
            f.write(text)
        return "ok"

    def _run(self, cmd: list[str], timeout: int = 300) -> str:
        done = subprocess.run(cmd, cwd=self.root, env=STUB_ENV, capture_output=True, text=True,
                              timeout=timeout)
        out = (done.stdout + done.stderr).strip() or "(пустой вывод)"
        return f"{out[-4500:]}\n[код возврата {done.returncode}]"

    def run_python(self, code: str) -> str:
        return self._run([str(self.root / ".venv/bin/python"), "-c", code], 60)

    def run_pytest(self, args: str = "tests/ -q") -> str:
        return self._run([str(self.root / ".venv/bin/pytest"), "-p", "no:cacheprovider",
                          *shlex.split(args)])

    def run_ruff(self, paths: str = "src") -> str:
        return self._run([str(self.root / ".venv/bin/ruff"), "check", "--output-format", "concise",
                          *shlex.split(paths)], 120)

    def run_compileall(self) -> str:
        python = str(self.root / ".venv/bin/python")
        return self._run([python, "-m", "compileall", "-q", "src"], 120)

    def git_diff(self) -> str:
        return self._run(["git", "diff", "--no-ext-diff", "--no-textconv"], 30)


def _tool(name: str, desc: str, props: dict[str, str], required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "required": required,
                       "properties": {k: {"type": v} for k, v in props.items()}}}}


SCHEMAS = {s["function"]["name"]: s for s in [
    _tool("list_dir", "Список файлов каталога", {"path": "string"}, ["path"]),
    _tool("read_file", "Прочитать файл с номерами строк (до 250 строк)",
          {"path": "string", "start_line": "integer", "end_line": "integer"}, ["path"]),
    _tool("grep", "Поиск регулярным выражением по файлам", {"pattern": "string", "path": "string"},
          ["pattern"]),
    _tool("edit_file", "Заменить ровно один фрагмент файла в src/ или tests/",
          {"path": "string", "old_string": "string", "new_string": "string"},
          ["path", "old_string", "new_string"]),
    _tool("append_file", "Дописать текст в конец файла в src/ или tests/",
          {"path": "string", "text": "string"}, ["path", "text"]),
    _tool("run_python", "Выполнить Python-код (python -c, PYTHONPATH=src)", {"code": "string"},
          ["code"]),
    _tool("run_pytest", "pytest с аргументами, например 'tests/test_x.py -q'",
          {"args": "string"}, []),
    _tool("run_ruff", "ruff check по путям через пробел", {"paths": "string"}, []),
    _tool("run_compileall", "python -m compileall -q src", {}, []),
    _tool("git_diff", "Показать git diff рабочего дерева", {}, []),
]}


def system_message(profile: Profile, variant: str, root: Path) -> str:
    """Тело профиля выбранного варианта + строка про инструменты."""
    if variant == "v2":
        source = SCRIPTS_ROOT / ".continue" / "rules" / profile.rule_file
    else:
        source = root / ".claude" / "agents" / profile.agent_file
    body = lib.profile_body(source.read_text(encoding="utf-8"))
    return f"{body}\n\n## Инструменты в этой среде\n{profile.tools_note}"


def agent_loop(args: argparse.Namespace, profile: Profile, ws: Workspace, system: str) -> dict:
    """Цикл «модель → инструменты → модель» до ответа без вызовов или лимита."""
    tools = [SCHEMAS[name] for name in profile.tools]
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": args.task or profile.task}]
    think = json.loads(args.think) if args.think else lib.default_think(args.model)
    log: list[dict] = []
    stats = {"gen": 0, "eval_ns": 0, "prefill_ns": 0, "max_prompt": 0, "model_errors": 0}
    last_error = ""
    final, reason, turn = None, "max_turns", 0

    def ask(with_tools: bool) -> dict:
        body: dict[str, Any] = {
            "model": args.model, "messages": messages,
            "options": {"num_ctx": NUM_CTX, "temperature": 0.2, "top_p": 0.9, "num_predict": 4096},
        }
        if with_tools:
            body["tools"] = tools
        if think is not None:
            body["think"] = think
        for attempt in range(MODEL_RETRIES + 1):
            try:
                data = client.chat(args.url, body)
                break
            except client.OllamaError as exc:
                stats["model_errors"] += 1
                if exc.status < 500 or attempt == MODEL_RETRIES:
                    raise
        stats["gen"] += data["eval_count"]
        stats["eval_ns"] += data["eval_duration"]
        stats["prefill_ns"] += data["prompt_eval_duration"]
        stats["max_prompt"] = max(stats["max_prompt"], data["prompt_eval_count"])
        return data

    for turn in range(1, profile.max_turns + 1):
        try:
            data = ask(True)
        except client.OllamaError as exc:
            final, reason, last_error = None, "model_error", str(exc)
            break
        reply = data["message"]
        messages.append({k: v for k, v in reply.items()
                         if k in ("role", "content", "tool_calls", "thinking")})
        if not reply.get("tool_calls"):
            final, reason = reply["content"], "answer"
            break
        for call in reply["tool_calls"]:
            name, call_args = call["function"]["name"], call["function"]["arguments"]
            method = getattr(ws, name, None) if name in profile.tools else None
            try:
                result = method(**call_args) if method else f"ошибка: инструмента {name} нет"
            except Exception as exc:  # noqa: BLE001 — ошибка инструмента возвращается модели
                result = f"ошибка вызова: {exc!r}"
            shown = {k: (v[:200] if isinstance(v, str) else v) for k, v in call_args.items()}
            log.append({"turn": turn, "tool": name, "args": shown,
                        "result_head": result[:200].replace("\n", " ")})
            messages.append({"role": "tool", "tool_name": name, "content": result})
        if data["prompt_eval_count"] > TRIM_AT:
            lib.trim_tool_results(messages)
        if data["prompt_eval_count"] > HARD_STOP:
            reason = "context_limit"
            break
    if final is None:
        messages.append({"role": "user", "content": (
            "Лимит исчерпан. Дай итоговый ответ в формате профиля по тому, что уже найдено или "
            "сделано; чего не проверил — пометь.")})
        try:
            final = ask(False)["message"]["content"]
        except client.OllamaError as exc:
            final, last_error = f"(модель не дала итогового ответа: {exc})", str(exc)
    return {"answer": final, "log": log, "stop_reason": reason, "turns": turn,
            "last_error": last_error, **stats}


def git_status(root: Path) -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                          text=True).stdout


def research_checks(root: Path, answer: str) -> dict:
    """Ссылки «файл:строка» и доля непокрытых модулей, названных в ответе."""
    refs, valid = lib.find_refs(answer), 0
    for name, line in refs:
        path = root / name
        if not path.is_file():
            found = list(root.glob(f"src/**/{name}")) + list(root.glob(f"tests/{name}"))
            path = found[0] if len(found) == 1 else path
        if path.is_file():
            total = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            valid += lib.ref_is_valid(line, total)
    modules = [str(p.relative_to(root / "src")) for p in (root / "src").rglob("*.py")
               if p.relative_to(root / "src").parts[0] not in RESEARCH_EXCLUDED]
    tests = "\n".join(p.read_text(encoding="utf-8") for p in (root / "tests").glob("*.py"))
    truth = lib.uncovered_modules(modules, tests)
    named = [m for m in truth
             if m in answer or (Path(m).name not in GENERIC_NAMES and Path(m).name in answer)]
    return {"refs_total": len(refs), "refs_valid": valid, "uncovered_truth": len(truth),
            "uncovered_named": len(named)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", choices=sorted(PROFILES))
    parser.add_argument("root", type=Path, help="корень проекта (bugfix — одноразовый worktree)")
    parser.add_argument("out", type=Path)
    parser.add_argument("model")
    parser.add_argument("--variant", choices=["v1", "v2"], default="v2")
    parser.add_argument("--think", help='JSON для параметра think; по умолчанию — по модели')
    parser.add_argument("--task", help="свой текст задачи вместо стандартного")
    parser.add_argument("--url", default=client.DEFAULT_URL)
    args = parser.parse_args()

    profile, root = PROFILES[args.profile], args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    before = git_status(root)
    started = time.time()
    result = agent_loop(args, profile, Workspace(root, profile.writable),
                        system_message(profile, args.variant, root))
    calls: dict[str, int] = {}
    for entry in result["log"]:
        calls[entry["tool"]] = calls.get(entry["tool"], 0) + 1
    metrics = {
        "profile": args.profile, "variant": args.variant, "model": args.model,
        "stop_reason": result["stop_reason"], "turns": result["turns"], "tool_calls": calls,
        "tool_errors": sum(e["result_head"].startswith("ошибка") for e in result["log"]),
        "wall_s": round(time.time() - started, 1), "gen_tokens": result["gen"],
        "gen_tps": lib.rate(result["gen"], result["eval_ns"]),
        "prefill_total_s": round(result["prefill_ns"] / 1e9, 1),
        "max_prompt_tokens": result["max_prompt"], "repo_changed": git_status(root) != before,
        "model_errors": result["model_errors"], "last_error": result["last_error"],
    }
    if args.profile == "research":
        metrics.update(research_checks(root, result["answer"]))
    (args.out / "answer.md").write_text(result["answer"], encoding="utf-8")
    (args.out / "toolcalls.json").write_text(
        json.dumps(result["log"], ensure_ascii=False, indent=1), encoding="utf-8")
    diff = subprocess.run(["git", "diff", "--no-ext-diff"], cwd=root, capture_output=True,
                          text=True)
    (args.out / "final.diff").write_text(diff.stdout, encoding="utf-8")
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
