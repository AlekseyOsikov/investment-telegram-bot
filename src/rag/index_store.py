"""Хранение индекса FAISS+SQLite по каждой стратегии чанкинга отдельно (rag/cli.py) —
см. Requirement «Хранение индекса по стратегиям» в
openspec/changes/add-rag-indexing-pipeline/specs/rag-indexing/spec.md.

Файлы одной стратегии — `<RAG_INDEX_DIR>/<strategy>/index.faiss` (косинусная близость
через `IndexFlatIP` по L2-нормализованным векторам — точный поиск, оправданный
масштабом корпуса, см. design.md, решение «Тип индекса FAISS») и
`<RAG_INDEX_DIR>/<strategy>/meta.sqlite3` (одна таблица `chunks`). `build_index()`
собирает оба файла в ОТДЕЛЬНОМ временном каталоге и атомарно (посредством `os.rename`
директории, а не построчной записи) подменяет ими предыдущие — крах посреди сборки
не может оставить прежний индекс частично переписанным (см. Requirement «Построение
эмбеддингов», сценарий «Сбой построения эмбеддинга не оставляет частичный индекс», и
Requirement «Хранение индекса по стратегиям», сценарий «Полная пересборка при
повторном запуске»).
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    date TEXT,
    strategy TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    faiss_row_id INTEGER NOT NULL
)
"""


@dataclass(frozen=True)
class IndexRecord:
    """Один чанк со своим эмбеддингом, готовый к сохранению — собирается вызывающим
    кодом (rag/cli.py) из FileMetadata (rag/filenames.py) + rag.chunking.Chunk +
    вектора providers.embeddings_client.embed_texts()."""

    source: str
    title: str
    author: str | None
    date: str | None
    chunk_index: int
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    """Один результат поиска — строка meta.sqlite3, найденная по FAISS, плюс её
    оценка близости (косинусное сходство, тем выше — тем ближе)."""

    chunk_id: str
    source: str
    title: str
    author: str | None
    date: str | None
    chunk_index: int
    text: str
    score: float


@dataclass(frozen=True)
class IndexStats:
    """Статистика по чанкам одной стратегии для /research_chunking_stats."""

    total_chunks: int
    avg_chars: float
    min_chars: int
    max_chars: int


def _strategy_dir(strategy: str, index_dir: str) -> str:
    return os.path.join(index_dir, strategy)


def _cleanup_stale_dirs(strategy: str, index_dir: str) -> None:
    """Лучшим усилием убирает `<strategy>.stale-*`, оставшиеся от НЕудавшейся уборки
    прошлых запусков build_index() (см. её докстринг про то, откуда они берутся —
    например, пока какой-то другой процесс на хосте держит открытым файл старой
    версии индекса). Вызывается в НАЧАЛЕ build_index(): если та внешняя блокировка к
    этому моменту снята, старый мусор наконец удаляется; если нет — не мешает
    текущей сборке, она с этими каталогами никак не взаимодействует."""
    pattern = os.path.join(index_dir, f"{strategy}.stale-*")
    for stale_dir in glob.glob(pattern):
        try:
            shutil.rmtree(stale_dir)
        except OSError:
            pass


def _normalize(vector: list[float]) -> np.ndarray:
    """L2-нормализует вектор для косинусного сходства через IndexFlatIP (скалярное
    произведение единичных векторов = косинус угла между ними)."""
    array = np.asarray(vector, dtype="float32")
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


def index_exists(strategy: str, index_dir: str) -> bool:
    """Построен ли индекс этой стратегии — используется командами бота, чтобы
    ответить понятным сообщением вместо падения, если CLI ещё не запускали (см.
    Requirement «Команда бота со статистикой чанкинга»/«Команда бота для сравнения
    стратегий чанкинга», сценарии «Индекс ещё не построен»)."""
    strategy_dir = _strategy_dir(strategy, index_dir)
    return os.path.isfile(os.path.join(strategy_dir, "index.faiss")) and os.path.isfile(
        os.path.join(strategy_dir, "meta.sqlite3")
    )


def build_index(strategy: str, records: list[IndexRecord], index_dir: str) -> int:
    """Строит индекс стратегии `strategy` С НУЛЯ из ПОЛНОГО списка записей текущего
    состояния `data/source/` и атомарно заменяет предыдущие файлы этой стратегии
    (независимо от файлов других стратегий). Возвращает число сохранённых чанков.

    `records` может быть пустым (пустой корпус — Requirement «Command-line indexing
    entry point», сценарий «Пустой каталог с исходниками обработан корректно») — в
    этом случае сохраняется пустой, но валидный индекс (0 строк, читаемый как «индекс
    построен, чанков нет», а не как «индекс не построен вовсе»).
    """
    os.makedirs(index_dir, exist_ok=True)
    final_dir = _strategy_dir(strategy, index_dir)
    tmp_dir = os.path.join(index_dir, f"{strategy}.tmp")
    _cleanup_stale_dirs(strategy, index_dir)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    dim = len(records[0].embedding) if records else 1
    index = faiss.IndexFlatIP(dim)

    db_path = os.path.join(tmp_dir, "meta.sqlite3")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        for faiss_row_id, record in enumerate(records):
            vector = _normalize(record.embedding)
            index.add(np.array([vector], dtype="float32"))
            chunk_id = f"{record.source}::{record.chunk_index}"
            conn.execute(
                "INSERT INTO chunks "
                "(chunk_id, source, title, author, date, strategy, chunk_index, text, "
                "faiss_row_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    record.source,
                    record.title,
                    record.author,
                    record.date,
                    strategy,
                    record.chunk_index,
                    record.text,
                    faiss_row_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    faiss.write_index(index, os.path.join(tmp_dir, "index.faiss"))

    # Замена директории целиком — ОБА os.rename() ниже переименовывают строго НА
    # НЕСУЩЕСТВУЮЩИЙ путь: переименование каталога поверх УЖЕ СУЩЕСТВУЮЩЕГО
    # каталога-назначения ненадёжно на смонтированных через FUSE файловых системах
    # (NTFS/ntfs-3g: `fuseblk`) — более ранний вариант (final_dir -> фиксированный
    # "*.old", затем tmp_dir -> final_dir) на такой файловой системе надёжно падал с
    # `OSError: [Errno 39] Directory not empty`, даже когда каталог-назначение был
    # пуст (воспроизведено на реальном отчёте об ошибке). Имя для отвода прежней
    # версии — с PID и меткой времени, а не фиксированное "*.old": фиксированное имя
    # само может столкнуться с НЕудалённым остатком предыдущей неудачной попытки
    # уборки (см. ниже).
    #
    # Удаление отведённой прежней версии (stale_dir) — ЛУЧШИЙ УСИЛИЕ, а не часть
    # атомарной операции: на том же `fuseblk` был замечен случай, когда файл внутри
    # неё не удалялся немедленно (FUSE подменяет ещё занятый файл на скрытый
    # `.fuse_hiddenNNN` вместо настоящего unlink), из-за чего `shutil.rmtree`
    # завершался тем же `ENOTEMPTY` (воспроизведено: сторонний процесс на хосте —
    # например, IDE, индексирующая/просматривающая файл `meta.sqlite3` — держала
    # его открытым, из-за чего FUSE не мог выполнить настоящий unlink). Раз НОВЫЙ
    # индекс (final_dir) к этому моменту уже успешно подменён, сбой уборки мусора
    # не должен ронять всю индексацию — только предупреждение в журнал; сам каталог
    # ещё раз попробует удалить _cleanup_stale_dirs() при СЛЕДУЮЩЕМ запуске (когда
    # блокировка снаружи, возможно, уже снята), либо его можно удалить вручную.
    stale_dir = None
    if os.path.exists(final_dir):
        stale_dir = os.path.join(index_dir, f"{strategy}.stale-{os.getpid()}-{int(time.time())}")
        os.rename(final_dir, stale_dir)

    os.rename(tmp_dir, final_dir)

    if stale_dir is not None:
        try:
            shutil.rmtree(stale_dir)
        except OSError:
            logger.warning(
                "Не удалось удалить прежнюю версию индекса %r после замены новой — "
                "можно удалить каталог вручную позже, на сам индекс это не влияет.",
                stale_dir,
            )

    return len(records)


def search(
    strategy: str, index_dir: str, query_vector: list[float], top_k: int
) -> list[SearchResult]:
    """Топ-k похожих чанков стратегии `strategy` по вектору запроса. Пустой список,
    если у стратегии пока нет ни одного чанка."""
    strategy_dir = _strategy_dir(strategy, index_dir)
    index = faiss.read_index(os.path.join(strategy_dir, "index.faiss"))
    if index.ntotal == 0:
        return []

    vector = _normalize(query_vector)
    k = min(top_k, index.ntotal)
    scores, row_ids = index.search(np.array([vector], dtype="float32"), k)

    conn = sqlite3.connect(os.path.join(strategy_dir, "meta.sqlite3"))
    conn.row_factory = sqlite3.Row
    try:
        results: list[SearchResult] = []
        for score, faiss_row_id in zip(scores[0], row_ids[0]):
            if faiss_row_id < 0:
                continue
            row = conn.execute(
                "SELECT * FROM chunks WHERE faiss_row_id = ?", (int(faiss_row_id),)
            ).fetchone()
            if row is None:
                continue
            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    title=row["title"],
                    author=row["author"],
                    date=row["date"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    score=float(score),
                )
            )
        return results
    finally:
        conn.close()


def get_stats(strategy: str, index_dir: str) -> IndexStats | None:
    """Статистика по чанкам стратегии из meta.sqlite3 — `None`, если индекс этой
    стратегии ещё не построен вовсе (см. index_exists)."""
    if not index_exists(strategy, index_dir):
        return None

    db_path = os.path.join(_strategy_dir(strategy, index_dir), "meta.sqlite3")
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*), AVG(LENGTH(text)), MIN(LENGTH(text)), MAX(LENGTH(text)) "
            "FROM chunks"
        ).fetchone()
    finally:
        conn.close()

    total_chunks = row[0] or 0
    if total_chunks == 0:
        return IndexStats(total_chunks=0, avg_chars=0.0, min_chars=0, max_chars=0)
    return IndexStats(
        total_chunks=total_chunks,
        avg_chars=float(row[1]),
        min_chars=int(row[2]),
        max_chars=int(row[3]),
    )
