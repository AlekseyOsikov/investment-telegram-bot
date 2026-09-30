"""CLI индексации локального корпуса документов (`python -m rag.cli`, `make index`) —
см. Requirement «Точка входа индексации из командной строки» в
openspec/changes/add-rag-indexing-pipeline/specs/rag-indexing/spec.md.

Обрабатывает КАЖДЫЙ файл под `data/source/pdf/*.pdf` и `data/source/txt/*.txt`:
разбирает метаданные из имени (rag/filenames.py), извлекает и очищает текст
(rag/extraction.py), строит чанки ОБЕИМИ стратегиями (rag/chunking.py), получает
эмбеддинги (providers/embeddings_client.py) и сохраняет по индексу FAISS+SQLite на
стратегию (rag/index_store.py). Полностью пересобирает оба индекса с нуля при каждом
запуске — не инкрементальный (design.md, решение «Поведение CLI: полная пересборка»).

Каждая стратегия обрабатывается (эмбеддинги -> сохранение) ПОЛНОСТЬЮ независимо от
другой и последовательно: если построение эмбеддингов для одной стратегии не удалось,
CLI останавливается ДО вызова rag.index_store.build_index() для неё — прежний (с
предыдущего запуска, если он был) индекс ЭТОЙ стратегии остаётся нетронутым, а индекс
уже успешно обработанной стратегии к этому моменту уже сохранён (см. Requirement
«Построение эмбеддингов», сценарий «Сбой построения эмбеддинга не оставляет частичный
индекс»).
"""

from __future__ import annotations

import glob
import os
import sys

from providers.embeddings_client import embed_texts
from rag.chunking import chunk_fixed_size, chunk_structural
from rag.extraction import clean_text, extract_text
from rag.filenames import FileMetadata, parse_filename
from rag.index_store import IndexRecord, build_index

from config import RAG_FIXED_CHUNK_CHARS, RAG_FIXED_CHUNK_OVERLAP, RAG_INDEX_DIR

SOURCE_PDF_DIR = "data/source/pdf"
SOURCE_TXT_DIR = "data/source/txt"

# Сколько текстов чанков уходит в ОДИН вызов embed_texts — простая защита от
# единственного чрезмерно большого запроса к Ollama на весь корпус разом.
_EMBED_BATCH_SIZE = 32

# Один элемент — ещё не embedding'нутый чанк одной стратегии одного документа.
_PendingChunk = tuple[str, FileMetadata, int, str]


def _list_source_files() -> tuple[list[str], list[str]]:
    """Возвращает (pdf_files, txt_files) — отсортированные списки путей, каждый может
    быть пустым (Requirement «Command-line indexing entry point», сценарий «Пустой
    каталог с исходниками обработан корректно»)."""
    pdf_files = sorted(glob.glob(os.path.join(SOURCE_PDF_DIR, "*.pdf")))
    txt_files = sorted(glob.glob(os.path.join(SOURCE_TXT_DIR, "*.txt")))
    return pdf_files, txt_files


def _embed_all(texts: list[str]) -> list[list[float]]:
    """Строит эмбеддинги пачками по _EMBED_BATCH_SIZE текстов, сохраняя порядок."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        vectors.extend(embed_texts(batch))
    return vectors


def _build_strategy_index(strategy: str, pending: list[_PendingChunk]) -> int:
    """Строит эмбеддинги и сохраняет индекс ОДНОЙ стратегии; поднимает исключение (не
    перехватывает) при сбое эмбеддингов — см. докстринг модуля."""
    texts = [text for _, _, _, text in pending]
    print(f"[{strategy}] чанков собрано: {len(texts)}. Строю эмбеддинги…")
    vectors = _embed_all(texts)
    records = [
        IndexRecord(
            source=filename,
            title=metadata.title,
            author=metadata.author,
            date=metadata.date,
            chunk_index=chunk_index,
            text=text,
            embedding=vector,
        )
        for (filename, metadata, chunk_index, text), vector in zip(pending, vectors)
    ]
    count = build_index(strategy, records, RAG_INDEX_DIR)
    print(f"[{strategy}] сохранено чанков: {count} -> {os.path.join(RAG_INDEX_DIR, strategy)}/")
    return count


def index_documents() -> dict[str, int]:
    """Индексирует весь корпус обеими стратегиями, возвращает
    {"fixed": число_чанков, "structural": число_чанков} после успешного завершения
    обеих. Поднимает исключение как есть при сбое (extraction/эмбеддинги/сохранение)
    — main() переводит его в сообщение об ошибке и код возврата."""
    pdf_files, txt_files = _list_source_files()
    print(f"Найдено файлов: PDF={len(pdf_files)}, TXT={len(txt_files)}")

    fixed_pending: list[_PendingChunk] = []
    structural_pending: list[_PendingChunk] = []

    for path in pdf_files + txt_files:
        filename = os.path.basename(path)
        metadata = parse_filename(filename)
        raw_text = extract_text(path)
        cleaned_text = clean_text(raw_text)

        for chunk in chunk_fixed_size(
            cleaned_text,
            chunk_chars=RAG_FIXED_CHUNK_CHARS,
            overlap=RAG_FIXED_CHUNK_OVERLAP,
        ):
            fixed_pending.append((filename, metadata, chunk.chunk_index, chunk.text))

        for chunk in chunk_structural(raw_text):
            structural_pending.append((filename, metadata, chunk.chunk_index, chunk.text))

    result: dict[str, int] = {}
    result["fixed"] = _build_strategy_index("fixed", fixed_pending)
    result["structural"] = _build_strategy_index("structural", structural_pending)
    return result


def main() -> None:
    try:
        result = index_documents()
    except Exception as exc:  # noqa: BLE001 — CLI: печатаем понятную ошибку, а не трейсбек
        print(f"❌ Индексация остановлена из-за ошибки: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Готово:", ", ".join(f"{strategy}={count}" for strategy, count in result.items()))


if __name__ == "__main__":
    main()
