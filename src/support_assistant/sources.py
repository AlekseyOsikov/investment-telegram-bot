"""Перечень документов для индекса поддержки (design.md изменения add-support-assistant, решение 6).
Фиксированный набор в коде: `README.md` и `docs/support/**/*.md`. Исключено всё остальное:
`CLAUDE.md`, спецификации, исходники, `data/`, `.venv`. Расширять перечень — решение отдельного
изменения.

Корень проекта определяется по расположению файла, а не из ввода пользователя."""

from __future__ import annotations

import os
from pathlib import Path

# src/support_assistant/sources.py → корень репозитория на два уровня выше `src/`.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

EXCLUDED_DIR_NAMES = frozenset(
    {".venv", "venv", "node_modules", ".git", "data", "__pycache__", ".pytest_cache", ".ruff_cache"}
)
SUPPORT_DOCS_PREFIX = "docs/support/"


def _matches(rel_path: str) -> bool:
    if rel_path == "README.md":
        return True
    return rel_path.startswith(SUPPORT_DOCS_PREFIX) and rel_path.endswith(".md")


def list_source_files(root: str = PROJECT_ROOT) -> list[str]:
    """Относительные пути (с `/`) документов для индекса, по алфавиту — порядок стабилен."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(name for name in dirnames if name not in EXCLUDED_DIR_NAMES)
        for filename in filenames:
            rel_path = f"{rel_dir}/{filename}".lstrip("/")
            if _matches(rel_path):
                found.append(rel_path)
    return sorted(found)
