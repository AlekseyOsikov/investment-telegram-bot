"""Тесты сборки индекса документации (dev_assistant/indexer.py) без сети и faiss: эмбеддинг и
хранилище подставляются функциями."""

import json
import os

import pytest

from dev_assistant import indexer

LONG = "Содержательный абзац документации проекта. " * 10


def make_tree(root):
    files = {
        "README.md": f"# Проект\n\n{LONG}\n",
        "CLAUDE.md": f"# Правила\n\n{LONG}\n\n## Очередь\n\n{LONG}\n",
        "src/agents/CLAUDE.md": "   \n",
        "openspec/specs/queue/spec.md": f"# Очередь\n\n{LONG}\n",
        "openspec/changes/add-x/proposal.md": "не должен попасть",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_collect_counts_files_chunks_and_skips(tmp_path):
    make_tree(tmp_path)
    collected = indexer.collect_chunks(str(tmp_path))
    assert collected.files == 3
    assert ("src/agents/CLAUDE.md", "пустой файл") in collected.skipped
    sources = {chunk.source for chunk in collected.chunks}
    assert sources == {"README.md", "CLAUDE.md", "openspec/specs/queue/spec.md"}
    assert all(chunk.title and chunk.text.startswith(chunk.title) for chunk in collected.chunks)


def test_unreadable_file_is_skipped_not_fatal(tmp_path):
    make_tree(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "bad.md").write_bytes(b"\xff\xfe\x00bad\x80")
    collected = indexer.collect_chunks(str(tmp_path))
    assert any(path == "docs/bad.md" for path, _ in collected.skipped)
    assert collected.files == 3


def test_build_stores_records_and_meta(tmp_path):
    make_tree(tmp_path)
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    stored = {}

    def store(strategy, records, directory):
        stored.update(strategy=strategy, records=records, directory=directory)
        return len(records)

    summary = indexer.build(
        str(index_dir),
        str(tmp_path),
        embed=lambda texts: [[float(len(t)), 1.0] for t in texts],
        store=store,
        record_factory=lambda **kw: kw,
    )
    assert stored["strategy"] == "markdown" and stored["directory"] == str(index_dir)
    assert summary.chunks == len(stored["records"]) > 0
    assert all(r["author"] is None and r["date"] is None for r in stored["records"])
    assert indexer.read_built_at(str(index_dir)) == summary.built_at
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["files"] == 3


def test_embedding_failure_leaves_store_and_meta_untouched(tmp_path):
    make_tree(tmp_path)
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    calls = []

    def failing_embed(texts):
        raise ConnectionError("ollama недоступна")

    with pytest.raises(ConnectionError):
        indexer.build(
            str(index_dir),
            str(tmp_path),
            embed=failing_embed,
            store=lambda *a: calls.append(a),
            record_factory=lambda **kw: kw,
        )
    assert calls == []
    assert not os.path.exists(index_dir / "meta.json")


def test_embed_all_batches_and_keeps_order():
    seen = []

    def embed(batch):
        seen.append(len(batch))
        return [[float(text)] for text in batch]

    texts = [str(i) for i in range(indexer.EMBED_BATCH_SIZE * 2 + 3)]
    vectors = indexer.embed_all(texts, embed)
    assert [v[0] for v in vectors] == [float(i) for i in range(len(texts))]
    assert seen == [indexer.EMBED_BATCH_SIZE, indexer.EMBED_BATCH_SIZE, 3]


def test_summary_text_lists_skipped():
    summary = indexer.BuildSummary(2, 9, (("a.md", "пустой файл"),), "2026-10-14T00:00:00+00:00")
    text = indexer.format_summary(summary, "data/dev_index")
    assert "файлов: 2" in text and "фрагментов: 9" in text and "a.md" in text


def test_read_built_at_without_meta_file_is_none(tmp_path):
    assert indexer.read_built_at(str(tmp_path)) is None


@pytest.mark.parametrize(
    "content",
    [
        "{не json",
        "[]",
        '"строка"',
        "{}",
        '{"built_at": ""}',
        '{"built_at": 5}',
        '{"built_at": null}',
    ],
)
def test_read_built_at_of_unusable_meta_is_none(tmp_path, content):
    (tmp_path / "meta.json").write_text(content, encoding="utf-8")
    assert indexer.read_built_at(str(tmp_path)) is None


def test_read_built_at_returns_the_stored_date(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps({"built_at": "2026-10-01 12:00:00"}), encoding="utf-8"
    )
    assert indexer.read_built_at(str(tmp_path)) == "2026-10-01 12:00:00"
