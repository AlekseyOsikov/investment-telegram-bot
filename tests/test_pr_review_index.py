"""Тесты сборки индекса ревью (scripts/pr_review_index.py): настоящие FAISS+SQLite из
rag.index_store на синтетических векторах, без сети и Ollama. Файлы и тексты — вымышленные."""

import pr_review_index as idx
import pytest

from rag.index_store import IndexRecord, build_index, index_exists, search

PY_SOURCE = '"""Модуль."""\n\n\ndef alpha(x):\n    return x\n'
DOC = "# Заголовок\n\n" + "Текст раздела. " * 30 + "\n"


def _write(root, rel_path, text):
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _embed(texts):
    return [[1.0, float(len(text) % 7), 0.5] for text in texts]


@pytest.fixture
def repo(tmp_path):
    _write(tmp_path, "src/mod.py", PY_SOURCE)
    _write(tmp_path, "README.md", DOC)
    _write(tmp_path, "src/empty.py", "\n")
    return tmp_path


def test_collect_chunks_python_markdown_and_skipped(repo):
    files = ["README.md", "src/empty.py", "src/missing.py", "src/mod.py"]
    collected = idx.collect_chunks(str(repo), files)
    assert collected.files == 2
    titles = [c.title for c in collected.chunks]
    assert "src/mod.py › alpha(x)" in titles
    assert any(t.startswith("README.md") for t in titles)
    skipped = dict(collected.skipped)
    assert skipped["src/empty.py"] == "пустой файл"
    assert skipped["src/missing.py"].startswith("не прочитан")


def test_build_creates_index_and_meta(repo, tmp_path):
    index_dir = str(tmp_path / "index")
    summary = idx.build(
        index_dir,
        str(repo),
        ["README.md", "src/mod.py"],
        "abc1234",
        embed=_embed,
        store=build_index,
        record_factory=IndexRecord,
    )
    assert index_exists("review", index_dir)
    assert summary.files == 2 and summary.chunks >= 2
    meta = idx.read_meta(index_dir)
    assert meta["commit"] == "abc1234"
    assert meta["files"] == 2 and meta["chunks"] == summary.chunks
    assert meta["built_at"]
    found = search("review", index_dir, [1.0, 0.0, 0.5], 3)
    assert found and all(r.source in {"README.md", "src/mod.py"} for r in found)


def test_failed_embedding_keeps_previous_index_and_meta(repo, tmp_path):
    index_dir = str(tmp_path / "index")
    idx.build(
        index_dir, str(repo), ["src/mod.py"], "old", embed=_embed,
        store=build_index, record_factory=IndexRecord,
    )  # fmt: skip
    before = search("review", index_dir, [1.0, 0.0, 0.5], 5)
    meta_before = idx.read_meta(index_dir)

    def broken(_texts):
        raise ConnectionError("сервер эмбеддингов недоступен")

    with pytest.raises(ConnectionError):
        idx.build(
            index_dir, str(repo), ["README.md", "src/mod.py"], "new", embed=broken,
            store=build_index, record_factory=IndexRecord,
        )  # fmt: skip
    assert search("review", index_dir, [1.0, 0.0, 0.5], 5) == before
    assert idx.read_meta(index_dir) == meta_before


def test_read_meta_missing_or_broken(tmp_path):
    assert idx.read_meta(str(tmp_path)) is None
    (tmp_path / "meta.json").write_text("{не json", encoding="utf-8")
    assert idx.read_meta(str(tmp_path)) is None


def test_format_summary_lists_skipped():
    skipped = (("src/x.py", "пустой файл"),)
    summary = idx.BuildSummary(2, 5, skipped, "2026-10-01T00:00:00+00:00", None)
    text = idx.format_summary(summary, "/idx")
    assert "файлов: 2" in text and "src/x.py: пустой файл" in text and "не указан" in text
