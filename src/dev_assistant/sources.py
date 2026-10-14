"""Перечень документов проекта для индекса ассистента разработчика (design.md изменения
add-dev-assistant-mode, решение 3). Фиксированный набор в коде: `README.md`, `CLAUDE.md` (корневой
и вложенные в `src/`), `docs/**/*.md` (если есть), `openspec/specs/**/spec.md`. Исключено:
`openspec/changes/**` (в том числе архив), `data/`, `.venv`, исходники. Расширять перечень —
решение отдельного изменения.

Корень проекта определяется по расположению файла, а не из ввода пользователя."""

from __future__ import annotations

import os
from pathlib import Path

# src/dev_assistant/sources.py → корень репозитория на два уровня выше `src/`.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# Каталоги, в которые обход не заходит ни при каких глобах.
EXCLUDED_DIR_NAMES = frozenset(
    {".venv", "venv", "node_modules", ".git", "data", "__pycache__", ".pytest_cache", ".ruff_cache"}
)
EXCLUDED_PREFIXES = ("openspec/changes",)


def _is_excluded(rel_path: str) -> bool:
    parts = rel_path.split("/")
    if any(part in EXCLUDED_DIR_NAMES for part in parts[:-1]):
        return True
    return any(
        rel_path == prefix or rel_path.startswith(prefix + "/") for prefix in EXCLUDED_PREFIXES
    )


def _matches(rel_path: str) -> bool:
    name = rel_path.rsplit("/", 1)[-1]
    if rel_path == "README.md" or rel_path == "CLAUDE.md":
        return True
    if rel_path.startswith("src/") and name == "CLAUDE.md":
        return True
    if rel_path.startswith("docs/") and name.endswith(".md"):
        return True
    return rel_path.startswith("openspec/specs/") and name == "spec.md"


def list_source_files(root: str = PROJECT_ROOT) -> list[str]:
    """Относительные пути (с `/`) документов для индекса, по алфавиту — порядок стабилен."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        # Не заходим в исключённые каталоги: ни `.venv`, ни архив изменений не обходятся.
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in EXCLUDED_DIR_NAMES
            and not _is_excluded(f"{rel_dir}/{name}/x".lstrip("/"))
        )
        for filename in filenames:
            rel_path = f"{rel_dir}/{filename}".lstrip("/")
            if _matches(rel_path) and not _is_excluded(rel_path):
                found.append(rel_path)
    return sorted(found)
