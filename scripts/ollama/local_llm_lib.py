"""Чистые правила оснастки «локальная модель как код-ассистент» (`scripts/ollama/*.py`, отчёт —
`docs/local-llm.md`): границы доступа инструментов агента, нарезка файлов и поиск, разбор и
применение ответа модели блоками SEARCH/REPLACE, обрезка контекста, проверка ссылок «файл:строка»
и эвристика «модуль не импортируется тестами». Модуль не обращается ни к сети, ни к LLM, ни к
диску (запись и чтение передаются функциями), поэтому ветвления проверяются тестами
(`tests/test_local_llm_lib.py`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

# --- границы доступа инструментов агента ---

# Имя, начинающееся с `.env` (включая `.env.example`), и перечисленные каталоги модели не отдаются.
DENIED_PREFIX = ".env"
DENIED_NAMES = frozenset(
    {"data", ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "reports", ".coverage"}
)
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".example", ".cfg", ".ini"}
)

MAX_READ_LINES = 250
MAX_READ_CHARS = 7000
MAX_GREP_HITS = 60
MAX_GREP_LINE = 160

# Начало заглушки, которой заменяют старые результаты инструментов.
TRIMMED_MARK = "[результат удалён из контекста"


def split_rel_path(rel: str) -> tuple[str, ...] | None:
    """Части относительного пути или None: абсолютный путь и `..` отклоняются."""
    if rel.startswith("/"):
        return None
    parts = tuple(p for p in PurePosixPath(rel).parts if p not in ("", "."))
    if ".." in parts:
        return None
    return parts


def is_denied(parts: tuple[str, ...]) -> bool:
    """Путь закрыт от модели: секреты, данные, служебные каталоги."""
    return any(p.startswith(DENIED_PREFIX) or p in DENIED_NAMES for p in parts)


def is_writable(parts: tuple[str, ...], writable_dirs: tuple[str, ...]) -> bool:
    """Запись разрешена только внутри перечисленных каталогов верхнего уровня."""
    return bool(parts) and parts[0] in writable_dirs


# --- инструменты чтения ---


def read_report(rel: str, text: str, start: int = 1, end: int = 200) -> str:
    """Фрагмент файла с номерами строк: не больше MAX_READ_LINES строк и MAX_READ_CHARS символов."""
    lines = text.splitlines()
    first = max(1, int(start))
    last = min(len(lines), int(end), first + MAX_READ_LINES - 1)
    body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(first, last + 1))
    return f"[{rel}, строки {first}-{last} из {len(lines)}]\n{body}"[:MAX_READ_CHARS]


def grep_text(rel: str, text: str, pattern: re.Pattern[str]) -> list[str]:
    """Совпадения в одном файле в виде `путь:строка:текст`."""
    return [
        f"{rel}:{n}:{line.strip()[:MAX_GREP_LINE]}"
        for n, line in enumerate(text.splitlines(), 1)
        if pattern.search(line)
    ]


def grep_report(hits: list[str]) -> str:
    """Текст ответа поиска: не больше MAX_GREP_HITS совпадений, без совпадений — явная фраза."""
    if not hits:
        return "совпадений нет"
    text = "\n".join(hits[:MAX_GREP_HITS])
    return text + f"\n[показано {MAX_GREP_HITS} совпадений]" if len(hits) > MAX_GREP_HITS else text


def replace_once(text: str, old: str, new: str) -> tuple[str | None, int]:
    """Замена ровно одного вхождения: (новый текст или None, найдено вхождений)."""
    count = text.count(old) if old else 0
    return (text.replace(old, new) if count == 1 else None), count


def trim_tool_results(messages: list[dict], keep: int = 3) -> None:
    """Заменить заглушкой все результаты инструментов, кроме `keep` последних (на месте)."""
    idx = [
        i
        for i, m in enumerate(messages)
        if m["role"] == "tool" and not m["content"].startswith(TRIMMED_MARK)
    ]
    for i in idx[: max(len(idx) - keep, 0)]:
        messages[i]["content"] = f"{TRIMMED_MARK}: {len(messages[i]['content'])} симв.]"


# --- ответ модели блоками (однопроходная генерация) ---

_FILE_RE = re.compile(r"=== FILE: (.+?) ===\n(.*?)\n=== END ===", re.S)
_EDIT_RE = re.compile(r"=== EDIT: (.+?) ===\n(.*?)\n=== END ===", re.S)
_SEARCH_RE = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S)


@dataclass(frozen=True)
class FileBlock:
    path: str
    content: str


@dataclass(frozen=True)
class EditBlock:
    path: str
    search: str
    replace: str


def strip_fence(text: str) -> str:
    """Снять одну внешнюю markdown-ограду вокруг содержимого файла (формат её не просил)."""
    lines = text.strip("\n").split("\n")
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return "\n".join(lines)


def parse_answer(answer: str) -> tuple[list[FileBlock], list[EditBlock]]:
    """Новые файлы и правки существующих из ответа модели."""
    files = [FileBlock(m.group(1).strip(), m.group(2)) for m in _FILE_RE.finditer(answer)]
    edits = [
        EditBlock(m.group(1).strip(), s.group(1), s.group(2))
        for m in _EDIT_RE.finditer(answer)
        for s in _SEARCH_RE.finditer(m.group(2))
    ]
    return files, edits


def apply_answer(
    answer: str,
    read: Callable[[str], str | None],
    write: Callable[[str, str], None],
) -> list[str]:
    """Применить блоки через переданные read/write; вернуть описания неудавшихся правок."""
    files, edits = parse_answer(answer)
    for block in files:
        write(block.path, strip_fence(block.content) + "\n")
    problems = []
    for edit in edits:
        new, count = replace_once(read(edit.path) or "", edit.search, edit.replace)
        if new is None:
            problems.append(f"{edit.path}: SEARCH найден {count} раз: {edit.search[:60]!r}")
        else:
            write(edit.path, new)
    return problems


# --- проверка ответа ---

_REF_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|md|toml|yaml|example)):(\d+)")


def find_refs(text: str) -> list[tuple[str, int]]:
    """Ссылки вида `путь/файл.py:123` в тексте ответа."""
    return [(m.group(1), int(m.group(2))) for m in _REF_RE.finditer(text)]


def ref_is_valid(line: int, total_lines: int) -> bool:
    """Номер строки укажет на существующую строку файла."""
    return 1 <= line <= total_lines


def is_imported(dotted: str, tests_text: str) -> bool:
    """Модуль `pkg.mod` импортируется в тестах: `from pkg.mod import` или `from pkg import mod`."""
    pkg, _, name = dotted.rpartition(".")
    patterns = [rf"(?:from|import)\s+{re.escape(dotted)}\b"]
    if pkg:
        head = rf"from\s+{re.escape(pkg)}\s+import"
        patterns += [rf"{head}[^\n]*\b{name}\b", rf"{head}\s*\([^)]*\b{name}\b"]
    return any(re.search(p, tests_text) for p in patterns)


def uncovered_modules(rel_paths: list[str], tests_text: str) -> list[str]:
    """Модули (`pkg/mod.py` от src/), которых не импортирует ни один тест; `__init__` пропущен."""
    result = []
    for rel in sorted(rel_paths):
        parts = PurePosixPath(rel).with_suffix("").parts
        if parts[-1] == "__init__":
            continue
        if not is_imported(".".join(parts), tests_text):
            result.append(rel)
    return result


# --- скорость и параметры запросов ---


def rate(count: int, duration_ns: int) -> float | None:
    """Токенов в секунду по счётчику и длительности Ollama (наносекунды)."""
    return round(count / (duration_ns / 1e9), 1) if duration_ns else None


def default_think(model: str) -> str | bool | None:
    """Параметр `think` по умолчанию: рассуждение gpt-oss не отключить, у qwen3 — отключаем."""
    if model.startswith("gpt-oss"):
        return "low"
    if model.startswith("qwen3:"):
        return False
    return None


def profile_body(text: str) -> str:
    """Тело файла профиля без frontmatter."""
    if text.startswith("---"):
        return text.split("---", 2)[2].strip()
    return text.strip()
