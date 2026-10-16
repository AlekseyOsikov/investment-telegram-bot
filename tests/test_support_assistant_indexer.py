"""Тесты сборки индекса поддержки (support_assistant/indexer.py) без сети и faiss: эмбеддинг и
хранилище подставляются функциями."""

import json
import os

import pytest

from support_assistant import indexer

LONG = "Содержательный абзац справки поддержки. " * 10


def make_tree(root):
    files = {
        "README.md": f"# Бот\n\n{LONG}\n",
        "docs/support/faq.md": f"# FAQ\n\n{LONG}\n\n## Очередь\n\n{LONG}\n",
        "docs/support/empty.md": "   \n",
        "CLAUDE.md": "не должен попасть",
        "openspec/specs/x/spec.md": "не должен попасть",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_collect_uses_only_support_sources(tmp_path):
    make_tree(tmp_path)
    collected = indexer.collect_support_chunks(str(tmp_path))
    assert collected.files == 2
    assert {c.source for c in collected.chunks} == {"README.md", "docs/support/faq.md"}
    assert ("docs/support/empty.md", "пустой файл") in collected.skipped


def test_build_stores_records_in_given_dir_with_meta(tmp_path):
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
    assert indexer.read_built_at(str(index_dir)) == summary.built_at
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["files"] == 2


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


def test_summary_text_mentions_directory():
    summary = indexer.BuildSummary(2, 9, (), "2026-10-17T00:00:00+00:00")
    text = indexer.format_summary(summary, "data/support_index")
    assert "файлов: 2" in text and "data/support_index" in text
