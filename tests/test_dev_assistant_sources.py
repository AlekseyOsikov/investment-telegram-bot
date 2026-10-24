"""Тест перечня документов индекса (dev_assistant/sources.py) на временном дереве файлов."""

from dev_assistant.sources import list_source_files


def _touch(root, rel):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# x\n", encoding="utf-8")


def test_includes_and_excludes(tmp_path):
    included = [
        "README.md",
        "CLAUDE.md",
        "src/agents/CLAUDE.md",
        "src/dev_assistant/CLAUDE.md",
        "docs/guide.md",
        "docs/deep/more.md",
        "openspec/specs/queue/spec.md",
    ]
    excluded = [
        "openspec/changes/add-x/proposal.md",
        "openspec/changes/add-x/specs/y/spec.md",
        "openspec/changes/archive/2026-01-01-old/specs/z/spec.md",
        "data/notes.md",
        "data/docs/CLAUDE.md",
        ".venv/lib/pkg/README.md",
        ".venv/lib/pkg/CLAUDE.md",
        "src/agents/smart_agent.py",
        "openspec/specs/queue/design.md",
        "notes.md",
        "tests/CLAUDE.md",
    ]
    for rel in included + excluded:
        _touch(tmp_path, rel)

    assert list_source_files(str(tmp_path)) == sorted(included)


def test_order_is_stable(tmp_path):
    for rel in ("docs/b.md", "docs/a.md", "README.md"):
        _touch(tmp_path, rel)
    first = list_source_files(str(tmp_path))
    assert first == list_source_files(str(tmp_path)) == sorted(first)


def test_missing_docs_dir_is_fine(tmp_path):
    _touch(tmp_path, "README.md")
    assert list_source_files(str(tmp_path)) == ["README.md"]


def test_is_excluded_by_directory_name_and_by_prefix():
    from dev_assistant.sources import _is_excluded

    assert _is_excluded("data/notes.md") is True
    assert _is_excluded("docs/.venv/lib/README.md") is True
    assert _is_excluded("openspec/changes") is True  # сам префикс
    assert _is_excluded("openspec/changes/add-x/proposal.md") is True
    assert _is_excluded("openspec/changes-old/readme.md") is False  # другой каталог с общим началом
    assert _is_excluded("openspec/specs/queue/spec.md") is False
    assert _is_excluded("README.md") is False
    assert _is_excluded("docs/data.md") is False  # имя файла не считается каталогом
