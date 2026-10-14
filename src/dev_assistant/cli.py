"""CLI сборки индекса документации проекта: `python -m dev_assistant.cli`, `make dev-index`.

Отдельный от `rag/cli.py` (тот жёстко читает `data/source/pdf|txt` и пишет в `RAG_INDEX_DIR`):
здесь источники — документы репозитория, каталог — `DEV_ASSISTANT_INDEX_DIR`. Нужен локальный
Ollama с `EMBEDDINGS_MODEL`."""

from __future__ import annotations

import sys

from config import DEV_ASSISTANT_INDEX_DIR
from providers.embeddings_client import embed_texts
from rag.index_store import IndexRecord, build_index

from . import indexer


def main() -> None:
    try:
        summary = indexer.build(
            DEV_ASSISTANT_INDEX_DIR,
            embed=embed_texts,
            store=build_index,
            record_factory=IndexRecord,
        )
    except Exception as exc:  # noqa: BLE001 — CLI: понятная ошибка вместо трейсбека
        print(
            f"❌ Сборка индекса остановлена из-за ошибки: {exc}\n"
            "Прежний индекс (если был) не изменён.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(indexer.format_summary(summary, DEV_ASSISTANT_INDEX_DIR))


if __name__ == "__main__":
    main()
