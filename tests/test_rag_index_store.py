"""Тесты хранения индекса (rag/index_store.py) на небольших синтетических векторах —
без реальных эмбеддингов и без сети, по образцу test_market_tools.py: сама сессия
FAISS/SQLite здесь настоящая (не фейк), но входные векторы — вручную заданные числа."""

from rag.index_store import (
    IndexRecord,
    build_index,
    get_stats,
    index_exists,
    list_titles,
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


def test_list_titles_returns_one_title_per_document(tmp_path):
    index_dir = str(tmp_path)
    records = [
        _record("b.txt", 0, "первый чанк b", [1.0, 0.0]),
        _record("b.txt", 1, "второй чанк b", [0.9, 0.1]),
        _record("a.txt", 0, "единственный чанк a", [0.0, 1.0]),
    ]
    build_index("fixed", records, index_dir)

    # один заголовок на документ (несколько чанков b.txt не дублируются), по алфавиту
    assert list_titles("fixed", index_dir) == ["a", "b"]


def test_list_titles_empty_when_index_missing_or_empty(tmp_path):
    index_dir = str(tmp_path)
    assert list_titles("fixed", index_dir) == []
    build_index("fixed", [], index_dir)
    assert list_titles("fixed", index_dir) == []


# --- Дополнение: уборка прежних версий, нормализация вектора, поиск по неполным данным -----------

import logging  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from rag import index_store  # noqa: E402


def test_cleanup_removes_only_stale_dirs_of_the_same_strategy(tmp_path):
    for name in ("fixed.stale-1-100", "fixed.stale-2-200", "structural.stale-3-300", "fixed"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "file.txt").write_text("x", encoding="utf-8")
    index_store._cleanup_stale_dirs("fixed", str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fixed", "structural.stale-3-300"]


def test_cleanup_ignores_failure_to_remove(tmp_path, monkeypatch):
    (tmp_path / "fixed.stale-1-100").mkdir()

    def locked(path):
        raise OSError("файл занят")

    monkeypatch.setattr(index_store.shutil, "rmtree", locked)
    index_store._cleanup_stale_dirs("fixed", str(tmp_path))
    assert (tmp_path / "fixed.stale-1-100").exists()


def test_build_removes_old_stale_dirs_at_start(tmp_path):
    (tmp_path / "fixed.stale-1-100").mkdir()
    build_index("fixed", [_record("a.txt", 0, "текст", [1.0, 0.0])], str(tmp_path))
    assert not (tmp_path / "fixed.stale-1-100").exists()
    assert index_exists("fixed", str(tmp_path))


def test_build_removes_leftover_tmp_dir_of_a_failed_earlier_run(tmp_path):
    leftover = tmp_path / "fixed.tmp"
    leftover.mkdir()
    (leftover / "мусор.txt").write_text("x", encoding="utf-8")
    build_index("fixed", [_record("a.txt", 0, "текст", [1.0, 0.0])], str(tmp_path))
    assert not leftover.exists()
    assert get_stats("fixed", str(tmp_path)).total_chunks == 1


def test_rebuild_survives_failure_to_remove_the_previous_version(tmp_path, monkeypatch, caplog):
    build_index("fixed", [_record("a.txt", 0, "старый", [1.0, 0.0])], str(tmp_path))

    def locked(path):
        raise OSError("файл занят")

    monkeypatch.setattr(index_store.shutil, "rmtree", locked)
    with caplog.at_level(logging.WARNING):
        count = build_index("fixed", [_record("b.txt", 0, "новый", [0.0, 1.0])], str(tmp_path))
    assert count == 1
    assert "Не удалось удалить прежнюю версию индекса" in caplog.text
    monkeypatch.undo()
    assert [r.source for r in search("fixed", str(tmp_path), [0.0, 1.0], 1)] == ["b.txt"]
    assert any(name.startswith("fixed.stale-") for name in os.listdir(tmp_path))


def test_normalize_gives_unit_vector_and_keeps_zero_vector():
    unit = index_store._normalize([3.0, 4.0])
    assert unit.dtype == np.float32
    assert np.allclose(unit, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(unit), 1.0)
    zero = index_store._normalize([0.0, 0.0])
    assert np.allclose(zero, [0.0, 0.0])


def test_search_ignores_vector_length_and_orders_by_cosine(tmp_path):
    index_dir = str(tmp_path)
    build_index(
        "fixed",
        [
            _record("near.txt", 0, "близкий", [1.0, 0.1]),
            _record("far.txt", 0, "дальний", [0.0, 1.0]),
        ],
        index_dir,
    )
    results = search("fixed", index_dir, [100.0, 0.0], 2)  # длина вектора не влияет
    assert [r.source for r in results] == ["near.txt", "far.txt"]
    assert results[0].score == pytest.approx(1.0 / (1.0 + 0.01) ** 0.5, abs=1e-4)
    assert results[0].score > results[1].score


def test_search_with_top_k_larger_than_index_returns_everything(tmp_path):
    index_dir = str(tmp_path)
    build_index("fixed", [_record("a.txt", 0, "текст", [1.0, 0.0])], index_dir)
    assert len(search("fixed", index_dir, [1.0, 0.0], 50)) == 1


def test_search_in_empty_index_is_empty(tmp_path):
    build_index("fixed", [], str(tmp_path))
    assert search("fixed", str(tmp_path), [1.0], 3) == []


def test_search_skips_vectors_without_metadata_row(tmp_path):
    index_dir = str(tmp_path)
    build_index(
        "fixed",
        [
            _record("a.txt", 0, "первый", [1.0, 0.0]),
            _record("b.txt", 0, "второй", [0.9, 0.1]),
        ],
        index_dir,
    )
    conn = sqlite3.connect(os.path.join(index_dir, "fixed", "meta.sqlite3"))
    conn.execute("DELETE FROM chunks WHERE source = 'a.txt'")
    conn.commit()
    conn.close()
    assert [r.source for r in search("fixed", index_dir, [1.0, 0.0], 2)] == ["b.txt"]


def test_search_with_zero_query_vector_still_returns_results(tmp_path):
    index_dir = str(tmp_path)
    build_index("fixed", [_record("a.txt", 0, "текст", [1.0, 0.0])], index_dir)
    results = search("fixed", index_dir, [0.0, 0.0], 1)
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.0)
