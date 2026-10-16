"""Сборка индекса AI-ревью PR (`scripts/pr_review.py index`, `make review-index`, спека
`openspec/specs/pr-ai-review/`): документация проекта (Markdown-чанкер `/dev`) и исходники
`src/`, `scripts/` (чанкер по `ast`) → эмбеддинги → `rag.index_store.build_index()` в ОТДЕЛЬНЫЙ
каталог `PR_REVIEW_INDEX_DIR`, стратегия `review`.

Файлы читаются из рабочего каталога ОСНОВНОЙ ветки (runner делает checkout `main`), перечень — по
`git ls-files` и маске `pr_review_lib.select_index_files`; содержимое PR сюда не попадает. Полная
пересборка при каждом запуске; атомарность даёт `build_index()`: сбой эмбеддингов случается ДО его
вызова, прежний индекс и `meta.json` остаются нетронутыми.

Модуль не импортирует `config` и клиент эмбеддингов: `embed`, `store` и `record_factory`
подставляются вызывающим кодом (как в `dev_assistant.indexer`), поэтому сборка проверяется на
синтетических векторах (`tests/test_pr_review_index.py`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import pr_review_lib as lib

from dev_assistant.chunking import chunk_markdown
from dev_assistant.indexer import embed_all, now_iso

META_FILENAME = "meta.json"


@dataclass(frozen=True)
class PendingChunk:
    source: str
    title: str
    chunk_index: int
    text: str


@dataclass
class Collected:
    chunks: list[PendingChunk] = field(default_factory=list)
    files: int = 0  # файлов, давших хотя бы один фрагмент
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (путь, причина)


@dataclass(frozen=True)
class BuildSummary:
    files: int
    chunks: int
    skipped: tuple[tuple[str, str], ...]
    built_at: str
    commit: str | None


def chunk_file(rel_path: str, text: str) -> list[PendingChunk]:
    """Фрагменты одного файла: `.py` — по функциям и классам, `.md` — по заголовкам."""
    if rel_path.endswith(".py"):
        pieces = lib.chunk_python_source(text, rel_path)
    else:
        pieces = chunk_markdown(text, rel_path)
    return [PendingChunk(rel_path, p.title, p.chunk_index, p.text) for p in pieces]


def collect_chunks(root: str, files: list[str]) -> Collected:
    """Читает файлы перечня и делит на фрагменты. Нечитаемые и пустые пропускаются с причиной —
    сборка не падает из-за одного файла."""
    result = Collected()
    for rel_path in files:
        full_path = os.path.join(root, *rel_path.split("/"))
        try:
            with open(full_path, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            result.skipped.append((rel_path, f"не прочитан ({type(exc).__name__})"))
            continue
        chunks = chunk_file(rel_path, text)
        if not chunks:
            result.skipped.append((rel_path, "пустой файл"))
            continue
        result.files += 1
        result.chunks.extend(chunks)
    return result


def meta_path(index_dir: str) -> str:
    return os.path.join(index_dir, META_FILENAME)


def write_meta(index_dir: str, summary: BuildSummary) -> None:
    """Метаданные сборки рядом с индексом; запись через временный файл и `os.replace`."""
    payload = {
        "built_at": summary.built_at,
        "commit": summary.commit,
        "files": summary.files,
        "chunks": summary.chunks,
        "skipped": [[path, reason] for path, reason in summary.skipped],
    }
    target = meta_path(index_dir)
    tmp = f"{target}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def read_meta(index_dir: str) -> dict | None:
    """Метаданные индекса или None (нет файла / не читается)."""
    try:
        with open(meta_path(index_dir), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def format_summary(summary: BuildSummary, index_dir: str) -> str:
    lines = [
        f"Проиндексировано файлов: {summary.files}, фрагментов: {summary.chunks} "
        f"→ {os.path.join(index_dir, lib.STRATEGY)}/",
        f"Дата сборки: {summary.built_at}; коммит: {summary.commit or 'не указан'}",
    ]
    lines.append(f"Пропущено файлов: {len(summary.skipped)}")
    lines.extend(f"  - {path}: {reason}" for path, reason in summary.skipped)
    return "\n".join(lines)


def build(
    index_dir: str,
    root: str,
    files: list[str],
    commit: str | None,
    *,
    embed: Callable[[list[str]], list[list[float]]],
    store: Callable[[str, list, str], int],
    record_factory: Callable[..., object],
) -> BuildSummary:
    """Полная пересборка индекса ревью. Исключение `embed` пробрасывается ДО `store`: прежний
    индекс цел, `meta.json` не обновляется."""
    collected = collect_chunks(root, files)
    vectors = embed_all([chunk.text for chunk in collected.chunks], embed)
    records = [
        record_factory(
            source=chunk.source,
            title=chunk.title,
            author=None,
            date=None,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            embedding=vector,
        )
        for chunk, vector in zip(collected.chunks, vectors, strict=True)
    ]
    store(lib.STRATEGY, records, index_dir)
    summary = BuildSummary(
        files=collected.files,
        chunks=len(records),
        skipped=tuple(collected.skipped),
        built_at=now_iso(),
        commit=commit,
    )
    write_meta(index_dir, summary)
    return summary
