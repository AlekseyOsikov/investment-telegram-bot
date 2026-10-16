"""Сборка индекса справки поддержки (`make support-index`, design.md изменения
add-support-assistant, решение 6): чтение `docs/support/**/*.md` и `README.md` → чанки по
заголовкам → эмбеддинги → `build_index` из `rag/` в ОТДЕЛЬНЫЙ каталог `SUPPORT_INDEX_DIR`. Полная
пересборка при каждом запуске; атомарность даёт `rag.index_store.build_index()` (сбой эмбеддингов —
до его вызова, прежний индекс цел). Индексы учебного корпуса и документации разработчика не
читаются и не меняются.

Чтение файлов, чанкер, эмбеддинг пачками и метаданные сборки — общие с `/dev`
(`dev_assistant.indexer`), поведение `/dev` не затрагивается; здесь только свой перечень файлов."""

from __future__ import annotations

from collections.abc import Callable

from dev_assistant.indexer import (
    STRATEGY,
    BuildSummary,
    Collected,
    collect_chunks,
    embed_all,
    format_summary,
    now_iso,
    read_built_at,
    write_meta,
)

from .sources import PROJECT_ROOT, list_source_files

__all__ = [
    "STRATEGY",
    "BuildSummary",
    "Collected",
    "build",
    "collect_support_chunks",
    "format_summary",
    "read_built_at",
]


def collect_support_chunks(root: str = PROJECT_ROOT) -> Collected:
    """Фрагменты документов поддержки (перечень — `sources.list_source_files`)."""
    return collect_chunks(root, files=list_source_files(root))


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
    collected = collect_support_chunks(root)
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
