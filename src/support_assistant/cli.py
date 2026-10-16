"""CLI сборки индекса справки поддержки: `python -m support_assistant.cli`, `make support-index`.

Отдельный от `rag/cli.py` и `dev_assistant/cli.py`: источники — `docs/support/` и `README.md`,
каталог — `SUPPORT_INDEX_DIR`. Нужен локальный Ollama с `EMBEDDINGS_MODEL`."""

from __future__ import annotations

import sys

from config import SUPPORT_INDEX_DIR
from providers.embeddings_client import embed_texts
from rag.index_store import IndexRecord, build_index

from . import indexer


def main() -> None:
    try:
        summary = indexer.build(
            SUPPORT_INDEX_DIR,
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
    print(indexer.format_summary(summary, SUPPORT_INDEX_DIR))


if __name__ == "__main__":
    main()
