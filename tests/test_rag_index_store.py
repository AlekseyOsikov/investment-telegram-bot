"""Тесты хранения индекса (rag/index_store.py) на небольших синтетических векторах —
без реальных эмбеддингов и без сети, по образцу test_market_tools.py: сама сессия
FAISS/SQLite здесь настоящая (не фейк), но входные векторы — вручную заданные числа."""

from rag.index_store import (
    IndexRecord,
    build_index,
    get_stats,
    index_exists,
    search,
)


def _record(source: str, chunk_index: int, text: str, embedding: list[float]) -> IndexRecord:
    return IndexRecord(
        source=source,
        title=source.rsplit(".", 1)[0],
        author=None,
        date=None,
        chunk_index=chunk_index,
        text=text,
        embedding=embedding,
    )


def test_index_does_not_exist_before_build(tmp_path):
    index_dir = str(tmp_path)
    assert index_exists("fixed", index_dir) is False
    assert get_stats("fixed", index_dir) is None


def test_independent_files_per_strategy(tmp_path):
    index_dir = str(tmp_path)
    records = [_record("a.txt", 0, "текст один", [1.0, 0.0])]
    build_index("fixed", records, index_dir)

    assert index_exists("fixed", index_dir) is True
    assert index_exists("structural", index_dir) is False  # другая стратегия не тронута

    build_index("structural", [_record("a.txt", 0, "другой текст", [0.0, 1.0])], index_dir)
    assert index_exists("structural", index_dir) is True

    fixed_stats = get_stats("fixed", index_dir)
    structural_stats = get_stats("structural", index_dir)
    assert fixed_stats.total_chunks == 1
    assert structural_stats.total_chunks == 1


def test_full_rebuild_replaces_previous_contents(tmp_path):
    index_dir = str(tmp_path)
    build_index(
        "fixed",
        [
            _record("a.txt", 0, "первая версия, чанк 1", [1.0, 0.0]),
            _record("a.txt", 1, "первая версия, чанк 2", [0.9, 0.1]),
        ],
        index_dir,
    )
    assert get_stats("fixed", index_dir).total_chunks == 2

    build_index("fixed", [_record("b.txt", 0, "новая версия", [0.0, 1.0])], index_dir)
    stats = get_stats("fixed", index_dir)
    assert stats.total_chunks == 1  # старые записи не остались рядом с новыми

    results = search("fixed", index_dir, [0.0, 1.0], top_k=5)
    assert len(results) == 1
    assert results[0].source == "b.txt"


def test_chunk_id_unique_within_strategy(tmp_path):
    index_dir = str(tmp_path)
    records = [
        _record("a.txt", 0, "чанк a-0", [1.0, 0.0]),
        _record("a.txt", 1, "чанк a-1", [0.9, 0.1]),
        _record("b.txt", 0, "чанк b-0", [0.0, 1.0]),
    ]
    build_index("fixed", records, index_dir)

    results = search("fixed", index_dir, [1.0, 0.0], top_k=10)
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_top_k_search_returns_correct_metadata(tmp_path):
    index_dir = str(tmp_path)
    records = [
        _record("close.txt", 0, "почти совпадает с запросом", [1.0, 0.0, 0.0]),
        _record("far.txt", 0, "совсем не похоже на запрос", [0.0, 1.0, 0.0]),
        _record("mid.txt", 0, "что-то среднее", [0.7, 0.7, 0.0]),
    ]
    build_index("fixed", records, index_dir)

    results = search("fixed", index_dir, [1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].source == "close.txt"
    assert results[0].text == "почти совпадает с запросом"
    # результаты отсортированы по убыванию похожести
    assert results[0].score >= results[1].score


def test_stats_correctness(tmp_path):
    index_dir = str(tmp_path)
    records = [
        _record("a.txt", 0, "12345", [1.0, 0.0]),  # длина 5
        _record("a.txt", 1, "1234567890", [0.9, 0.1]),  # длина 10
    ]
    build_index("fixed", records, index_dir)

    stats = get_stats("fixed", index_dir)
    assert stats.total_chunks == 2
    assert stats.min_chars == 5
    assert stats.max_chars == 10
    assert stats.avg_chars == 7.5


def test_build_with_empty_records_is_valid_empty_index(tmp_path):
    index_dir = str(tmp_path)
    count = build_index("fixed", [], index_dir)
    assert count == 0
    assert index_exists("fixed", index_dir) is True
    stats = get_stats("fixed", index_dir)
    assert stats.total_chunks == 0
    assert search("fixed", index_dir, [1.0], top_k=5) == []
