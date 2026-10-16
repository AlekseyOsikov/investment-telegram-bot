"""Тест перечня документов индекса поддержки (support_assistant/sources.py) на временном дереве, а
также проверка реального FAQ репозитория: читается чанкером и несёт оговорку о советнике."""

from pathlib import Path

from dev_assistant.chunking import chunk_markdown
from support_assistant.sources import PROJECT_ROOT, list_source_files


def _touch(root, rel, text="# x\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_includes_and_excludes(tmp_path):
    included = ["README.md", "docs/support/faq.md", "docs/support/deep/more.md"]
    excluded = [
        "CLAUDE.md",
        "src/agents/CLAUDE.md",
        "docs/other/guide.md",
        "docs/support/notes.txt",
        "openspec/specs/queue/spec.md",
        "openspec/changes/add-x/proposal.md",
        "data/docs/support/x.md",
        ".venv/lib/pkg/README.md",
        "src/main.py",
        "tests/README.md",
    ]
    for rel in included + excluded:
        _touch(tmp_path, rel)
    assert list_source_files(str(tmp_path)) == sorted(included)


def test_missing_tree_gives_empty_list(tmp_path):
    assert list_source_files(str(tmp_path)) == []


def test_real_support_docs_are_listed_and_chunkable():
    files = list_source_files(PROJECT_ROOT)
    support = [f for f in files if f.startswith("docs/support/")]
    assert "README.md" in files and support
    assert not any(f.endswith("CLAUDE.md") or f.startswith("openspec/") for f in files)
    for rel in support:
        text = (Path(PROJECT_ROOT) / rel).read_text(encoding="utf-8")
        assert chunk_markdown(text, rel), rel


def test_real_support_docs_carry_advisor_disclaimer():
    texts = " ".join(
        (Path(PROJECT_ROOT) / rel).read_text(encoding="utf-8")
        for rel in list_source_files(PROJECT_ROOT)
        if rel.startswith("docs/support/")
    )
    assert "не является лицензированным финансовым советником" in texts
    assert "гарантированная доходность" not in texts.lower()
