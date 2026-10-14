"""Сборка индекса документации проекта (`make dev-index`, design.md изменения
add-dev-assistant-mode, решение 3): чтение файлов → чанки по заголовкам → эмбеддинги → `build_index`
из `rag/` в ОТДЕЛЬНЫЙ каталог `DEV_ASSISTANT_INDEX_DIR`. Полная пересборка при каждом запуске;
атомарность даёт `rag.index_store.build_index()` (сбой эмбеддингов — до его вызова, прежний индекс
цел). Индекс инвестиционного корпуса (`RAG_INDEX_DIR`) не читается и не меняется.

Чистая часть (`collect_chunks`, `format_summary`, метаданные) тестируется без сети; сетевая
(`build`) принимает `embed` и `store` параметрами."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .chunking import chunk_markdown
from .sources import PROJECT_ROOT, list_source_files

STRATEGY = "markdown"
META_FILENAME = "meta.json"
EMBED_BATCH_SIZE = 16


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


def collect_chunks(root: str = PROJECT_ROOT, files: list[str] | None = None) -> Collected:
    """Читает документы и делит на фрагменты. Нечитаемые и пустые файлы пропускаются с
    причиной — сборка не падает из-за одного файла."""
    result = Collected()
    for rel_path in files if files is not None else list_source_files(root):
        full_path = os.path.join(root, *rel_path.split("/"))
        try:
            with open(full_path, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            result.skipped.append((rel_path, f"не прочитан ({type(exc).__name__})"))
            continue
        chunks = chunk_markdown(text, rel_path)
        if not chunks:
            result.skipped.append((rel_path, "пустой файл"))
            continue
        result.files += 1
        result.chunks.extend(
            PendingChunk(rel_path, chunk.title, chunk.chunk_index, chunk.text) for chunk in chunks
        )
    return result


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def meta_path(index_dir: str) -> str:
    return os.path.join(index_dir, META_FILENAME)


def write_meta(index_dir: str, summary: BuildSummary) -> None:
    """Метаданные сборки (дата, счётчики) рядом с индексом; запись через временный файл."""
    payload = {
        "built_at": summary.built_at,
        "files": summary.files,
        "chunks": summary.chunks,
        "skipped": [[path, reason] for path, reason in summary.skipped],
    }
    target = meta_path(index_dir)
    tmp = f"{target}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def read_built_at(index_dir: str) -> str | None:
    """Дата сборки индекса или None (нет метаданных / не читаются)."""
    try:
        with open(meta_path(index_dir), encoding="utf-8") as handle:
            value = json.load(handle).get("built_at")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value else None


def format_summary(summary: BuildSummary, index_dir: str) -> str:
    lines = [
        f"Проиндексировано файлов: {summary.files}, фрагментов: {summary.chunks} "
        f"→ {os.path.join(index_dir, STRATEGY)}/",
        f"Дата сборки: {summary.built_at}",
    ]
    if summary.skipped:
        lines.append(f"Пропущено файлов: {len(summary.skipped)}")
        lines.extend(f"  - {path}: {reason}" for path, reason in summary.skipped)
    else:
        lines.append("Пропущено файлов: 0")
    return "\n".join(lines)


def embed_all(
    texts: list[str], embed: Callable[[list[str]], list[list[float]]]
) -> list[list[float]]:
    """Эмбеддинги пачками, порядок сохраняется. Исключение пробрасывается: сборка
    останавливается ДО замены индекса."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(embed(texts[start : start + EMBED_BATCH_SIZE]))
    return vectors


def build(
    index_dir: str,
    root: str = PROJECT_ROOT,
    *,
    embed: Callable[[list[str]], list[list[float]]],
    store: Callable[[str, list, str], int],
    record_factory: Callable[..., object],
) -> BuildSummary:
    """Полная пересборка. `embed`, `store` (`rag.index_store.build_index`) и `record_factory`
    (`rag.index_store.IndexRecord`) подставляются вызывающим кодом: модуль не тянет faiss и
    клиент эмбеддингов при импорте и проверяется без них."""
    collected = collect_chunks(root)
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
    store(STRATEGY, records, index_dir)
    summary = BuildSummary(
        files=collected.files,
        chunks=len(records),
        skipped=tuple(collected.skipped),
        built_at=now_iso(),
    )
    write_meta(index_dir, summary)
    return summary
