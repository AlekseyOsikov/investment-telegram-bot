"""Тесты чистых правил сервера файлов проекта (mcp_servers/project_files/files_rules.py): пути,
denylist/allowlist, лимиты, diff, хранилище предложений. На временном каталоге, без сети и MCP."""

import json
import os
from dataclasses import replace

import files_rules as fr
import pytest
from files_rules import FilesError, Limits, ProposalStore

LIMITS = Limits()


def make_root(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "data").mkdir()
    (root / ".git").mkdir()
    (root / "README.md").write_text("# Проект\n\nсвязь\n", encoding="utf-8")
    (root / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (root / ".env").write_text("SECRET_TOKEN=abc123\n", encoding="utf-8")
    (root / ".env.example").write_text("TOKEN=\n", encoding="utf-8")
    (root / "data" / "dump.txt").write_text("SECRET_TOKEN\n", encoding="utf-8")
    return root


def code_of(call):
    with pytest.raises(FilesError) as info:
        call()
    return info.value.code


# --- пути -------------------------------------------------------------------------------------


def test_resolve_plain_and_root(tmp_path):
    root = make_root(tmp_path)
    assert fr.resolve_path(root, "README.md")[1] == "README.md"
    assert fr.resolve_path(root, "")[1] == ""
    assert fr.resolve_path(root, "./src/../README.md")[1] == "README.md"


def test_traversal_and_absolute_outside_rejected(tmp_path):
    root = make_root(tmp_path)
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    assert code_of(lambda: fr.resolve_path(root, "../outside.txt")) == fr.CODE_OUTSIDE
    assert code_of(lambda: fr.resolve_path(root, str(tmp_path / "outside.txt"))) == fr.CODE_OUTSIDE


def test_absolute_path_inside_root_is_allowed(tmp_path):
    root = make_root(tmp_path)
    assert fr.resolve_path(root, str(root / "README.md"))[1] == "README.md"


def test_symlink_outside_rejected(tmp_path):
    root = make_root(tmp_path)
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    os.symlink(tmp_path / "outside.txt", root / "link.txt")
    assert code_of(lambda: fr.resolve_path(root, "link.txt")) == fr.CODE_OUTSIDE


def test_symlink_to_env_rejected(tmp_path):
    root = make_root(tmp_path)
    os.symlink(root / ".env", root / "innocent.txt")
    assert code_of(lambda: fr.resolve_path(root, "innocent.txt")) == fr.CODE_DENIED


def test_denied_message_does_not_reveal_content(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(FilesError) as info:
        fr.resolve_path(root, ".env")
    assert "abc123" not in info.value.message and "SECRET" not in info.value.message


def test_nul_in_path_rejected(tmp_path):
    root = make_root(tmp_path)
    assert code_of(lambda: fr.resolve_path(root, "a\x00b")) == fr.CODE_BAD_ARGUMENT


@pytest.mark.parametrize(
    "path",
    [
        ".env", ".env.local", "sub/.env.prod", "data", "data/x.txt", ".git/config",
        ".venv/lib/x.py", ".venv311/a", "pkg/__pycache__/m.pyc", ".pytest_cache/v",
        "key.pem", "deploy/server.key", "id_rsa", "id_rsa.pub", "app.sqlite", "x.sqlite3-wal",
    ],
)
def test_read_denied(path):
    assert fr.is_read_denied(path)


@pytest.mark.parametrize(
    "path",
    [".env.example", "README.md", "src/main.py", "docs/a.md", "tests/data_loader.py", "mydata/x"],
)
def test_read_allowed(path):
    assert not fr.is_read_denied(path)


# --- allowlist записи -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "README.md", "CLAUDE.md", "src/dev_assistant/CLAUDE.md", "src/CLAUDE.md", "docs/a.md",
        "docs/adr/0001.md", ".env.example", "CHANGELOG.md",
    ],
)
def test_write_allowed(path):
    assert fr.is_write_allowed(path)


@pytest.mark.parametrize(
    "path", ["src/main.py", "src/config.py", ".env", "data/x.md", "tests/a.md", "OTHER.md", ""]
)
def test_write_denied(path):
    assert not fr.is_write_allowed(path)


def test_write_allow_extension_still_respects_read_denylist():
    assert fr.is_write_allowed("notes/a.md", ["notes/**"])
    assert not fr.is_write_allowed("data/a.md", ["data/**"])
    assert not fr.is_write_allowed(".env", [".env"])


# --- чтение, список, поиск --------------------------------------------------------------------


def test_read_numbered_range_and_total(tmp_path):
    root = make_root(tmp_path)
    text = fr.read_text(root / "README.md", LIMITS)
    result = fr.read_numbered(text, 2, 3, LIMITS)
    assert result["content"] == "2: \n3: связь"
    assert result["total_lines"] == 3 and not result["truncated"]


def test_read_numbered_truncation_marked():
    text = "\n".join(f"строка {i}" for i in range(100))
    result = fr.read_numbered(text, None, None, replace(LIMITS, max_read_chars=60))
    assert result["truncated"] and result["end_line"] < 100


def test_read_numbered_bad_range():
    assert code_of(lambda: fr.read_numbered("a\nb", 3, 2, LIMITS)) == fr.CODE_BAD_ARGUMENT


def test_read_binary_and_large_rejected(tmp_path):
    root = make_root(tmp_path)
    (root / "bin.dat").write_bytes(b"\x00\x01\x02")
    (root / "latin.txt").write_bytes("привет".encode("cp1251"))
    (root / "big.txt").write_text("x" * 100, encoding="utf-8")
    assert code_of(lambda: fr.read_text(root / "bin.dat", LIMITS)) == fr.CODE_BINARY
    assert code_of(lambda: fr.read_text(root / "latin.txt", LIMITS)) == fr.CODE_BINARY
    small = replace(LIMITS, max_file_bytes=50)
    assert code_of(lambda: fr.read_text(root / "big.txt", small)) == fr.CODE_TOO_LARGE
    assert code_of(lambda: fr.read_text(root / "none.txt", LIMITS)) == fr.CODE_NOT_FOUND
    assert code_of(lambda: fr.read_text(root / "src", LIMITS)) == fr.CODE_NOT_FILE


def test_list_directory_hides_denied(tmp_path):
    root = make_root(tmp_path)
    names = fr.list_directory(root, root, LIMITS)["entries"]
    assert "README.md" in names and "src/" in names and ".env.example" in names
    assert ".env" not in names and "data/" not in names and ".git/" not in names


def test_list_directory_hides_symlink_outside(tmp_path):
    root = make_root(tmp_path)
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    os.symlink(tmp_path / "outside.txt", root / "link.txt")
    assert "link.txt" not in fr.list_directory(root, root, LIMITS)["entries"]


def test_search_multiple_files_with_lines(tmp_path):
    root = make_root(tmp_path)
    (root / "docs" / "a.md").write_text("до\nuse FOO_SETTING here\n", encoding="utf-8")
    (root / "src" / "cfg.py").write_text("FOO_SETTING = 1\n", encoding="utf-8")
    result = fr.search(root, "FOO_SETTING", LIMITS)
    found = {(m["path"], m["line"]) for m in result["matches"]}
    assert found == {("docs/a.md", 2), ("src/cfg.py", 1)} and not result["truncated"]


def test_search_never_returns_denied_files(tmp_path):
    root = make_root(tmp_path)
    result = fr.search(root, "SECRET_TOKEN", LIMITS)
    assert result["matches"] == []


def test_search_skips_symlink_to_env(tmp_path):
    root = make_root(tmp_path)
    os.symlink(root / ".env", root / "innocent.txt")
    assert fr.search(root, "abc123", LIMITS)["matches"] == []


def test_search_truncates_at_limit_and_marks(tmp_path):
    root = make_root(tmp_path)
    (root / "docs" / "many.md").write_text("hit\n" * 10, encoding="utf-8")
    result = fr.search(root, "hit", replace(LIMITS, max_search_results=3))
    assert len(result["matches"]) == 3 and result["truncated"]


def test_search_exact_limit_not_truncated(tmp_path):
    root = make_root(tmp_path)
    (root / "docs" / "many.md").write_text("hit\n" * 3, encoding="utf-8")
    result = fr.search(root, "hit", replace(LIMITS, max_search_results=3))
    assert len(result["matches"]) == 3 and not result["truncated"]


def test_search_glob_regex_and_binary_skipped(tmp_path):
    root = make_root(tmp_path)
    (root / "docs" / "a.md").write_text("value=12\n", encoding="utf-8")
    (root / "src" / "a.py").write_text("value=34\n", encoding="utf-8")
    (root / "docs" / "bin.dat").write_bytes(b"value=\x00\x01")
    result = fr.search(root, r"value=\d+", LIMITS, path_glob="docs/**", regex=True)
    assert [m["path"] for m in result["matches"]] == ["docs/a.md"]
    assert result["skipped"] == 1


def test_search_bad_regex_and_empty_query(tmp_path):
    root = make_root(tmp_path)
    assert code_of(lambda: fr.search(root, "(", LIMITS, regex=True)) == fr.CODE_BAD_ARGUMENT
    assert code_of(lambda: fr.search(root, "", LIMITS)) == fr.CODE_BAD_ARGUMENT


# --- diff -------------------------------------------------------------------------------------


def test_diff_is_deterministic_and_git_format():
    one = fr.build_diff("README.md", "a\nb\n", "a\nc\n")
    two = fr.build_diff("README.md", "a\nb\n", "a\nc\n")
    assert one == two
    assert one.startswith("diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n")
    assert fr.count_changes(one) == (1, 1)


def test_diff_new_file_all_added():
    diff = fr.build_diff("docs/new.md", None, "x\ny\n")
    assert "new file mode 100644" in diff and "--- /dev/null" in diff
    assert fr.count_changes(diff) == (2, 0)


def test_diff_no_newline_marker():
    diff = fr.build_diff("README.md", "a\nb", "a\nb\n")
    assert "\\ No newline at end of file" in diff


# --- предложения ------------------------------------------------------------------------------


def propose(root, path, content, **kwargs):
    return fr.prepare_proposal(root, path, content, "сводка", LIMITS, **kwargs)


def test_propose_existing_does_not_touch_file(tmp_path):
    root = make_root(tmp_path)
    before = (root / "README.md").read_bytes()
    proposal = propose(root, "README.md", "# Проект\n\nновое\n")
    assert (root / "README.md").read_bytes() == before
    assert not proposal.creates and proposal.base_hash == fr.content_hash(before)
    assert proposal.added == 1 and proposal.removed == 1 and "новое" in proposal.diff


def test_propose_same_input_same_diff(tmp_path):
    root = make_root(tmp_path)
    one = propose(root, "README.md", "# Проект\n\nновое\n", proposal_id="a" * 12, now=1.0)
    two = propose(root, "README.md", "# Проект\n\nновое\n", proposal_id="b" * 12, now=2.0)
    assert one.diff == two.diff


def test_propose_new_file(tmp_path):
    root = make_root(tmp_path)
    proposal = propose(root, "docs/new.md", "# Новый\n")
    assert proposal.creates and proposal.base_hash is None
    assert proposal.added == 1 and proposal.removed == 0
    assert not (root / "docs" / "new.md").exists()


def test_propose_no_change_rejected(tmp_path):
    root = make_root(tmp_path)
    same = (root / "README.md").read_text(encoding="utf-8")
    assert code_of(lambda: propose(root, "README.md", same)) == fr.CODE_NO_CHANGE


def test_propose_source_code_not_writable(tmp_path):
    root = make_root(tmp_path)
    assert code_of(lambda: propose(root, "src/main.py", "x\n")) == fr.CODE_NOT_WRITABLE
    assert code_of(lambda: propose(root, ".env", "x\n")) == fr.CODE_DENIED
    assert code_of(lambda: propose(root, "../x.md", "x\n")) == fr.CODE_OUTSIDE


def test_propose_size_limit(tmp_path):
    root = make_root(tmp_path)
    small = replace(LIMITS, max_proposal_chars=10)
    assert code_of(
        lambda: fr.prepare_proposal(root, "README.md", "x" * 11, "", small)
    ) == fr.CODE_TOO_LARGE


def test_propose_crlf_file_rejected(tmp_path):
    root = make_root(tmp_path)
    (root / "docs" / "win.md").write_bytes(b"a\r\nb\r\n")
    assert code_of(lambda: propose(root, "docs/win.md", "a\nc\n")) == fr.CODE_CRLF


# --- хранилище и применение -------------------------------------------------------------------


def make_store(tmp_path, ttl=3600, clock=None):
    kwargs = {} if clock is None else {"clock": clock}
    return ProposalStore(tmp_path / "staging", ttl, **kwargs)


def test_store_roundtrip_and_double_take(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    proposal = propose(root, "README.md", "# Проект\n\nновое\n")
    store.save(proposal)
    assert store.count() == 1
    assert store.take(proposal.id) == proposal
    assert code_of(lambda: store.take(proposal.id)) == fr.CODE_UNKNOWN_PROPOSAL
    assert list((tmp_path / "staging").iterdir()) == []


def test_store_rejects_malformed_id(tmp_path):
    store = make_store(tmp_path)
    for bad in ("../x", "", "ABC", "a" * 40):
        assert code_of(lambda bad=bad: store.take(bad)) == fr.CODE_UNKNOWN_PROPOSAL
        assert code_of(lambda bad=bad: store.discard(bad)) == fr.CODE_UNKNOWN_PROPOSAL


def test_store_ttl_expired_removed(tmp_path):
    root = make_root(tmp_path)
    now = [1000.0]
    store = make_store(tmp_path, ttl=60, clock=lambda: now[0])
    proposal = propose(root, "README.md", "# Проект\n\nновое\n", now=1000.0)
    store.save(proposal)
    now[0] = 1061.0
    assert code_of(lambda: store.take(proposal.id)) == fr.CODE_EXPIRED
    assert store.count() == 0


def test_store_purge_on_save(tmp_path):
    root = make_root(tmp_path)
    now = [1000.0]
    store = make_store(tmp_path, ttl=60, clock=lambda: now[0])
    old = propose(root, "README.md", "# Проект\n\nA\n", now=1000.0, proposal_id="a" * 12)
    store.save(old)
    now[0] = 2000.0
    fresh = propose(root, "README.md", "# Проект\n\nB\n", now=2000.0, proposal_id="b" * 12)
    store.save(fresh)
    assert store.count() == 1


def test_store_discard(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    proposal = propose(root, "README.md", "# Проект\n\nновое\n")
    store.save(proposal)
    assert store.discard(proposal.id) is True
    assert store.discard(proposal.id) is False


def test_apply_writes_and_removes(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    proposal = propose(root, "README.md", "# Проект\n\nновое\n")
    store.save(proposal)
    applied = fr.apply_proposal(root, store, proposal.id)
    assert applied.path == "README.md"
    assert (root / "README.md").read_text(encoding="utf-8") == "# Проект\n\nновое\n"
    assert code_of(lambda: fr.apply_proposal(root, store, proposal.id)) == fr.CODE_UNKNOWN_PROPOSAL
    assert [p.name for p in root.iterdir() if p.suffix == ".tmp"] == []


def test_apply_creates_new_file_with_parents(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    proposal = propose(root, "docs/adr/0001.md", "# ADR\n")
    store.save(proposal)
    fr.apply_proposal(root, store, proposal.id)
    assert (root / "docs" / "adr" / "0001.md").read_text(encoding="utf-8") == "# ADR\n"


def test_apply_rejects_changed_source(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    proposal = propose(root, "README.md", "# Проект\n\nновое\n")
    store.save(proposal)
    (root / "README.md").write_text("кто-то поправил\n", encoding="utf-8")
    assert code_of(lambda: fr.apply_proposal(root, store, proposal.id)) == fr.CODE_CHANGED
    assert (root / "README.md").read_text(encoding="utf-8") == "кто-то поправил\n"


def test_apply_rejects_file_created_meanwhile(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    proposal = propose(root, "docs/new.md", "# Новый\n")
    store.save(proposal)
    (root / "docs" / "new.md").write_text("уже есть\n", encoding="utf-8")
    assert code_of(lambda: fr.apply_proposal(root, store, proposal.id)) == fr.CODE_EXISTS


def test_apply_rechecks_write_allowlist(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    proposal = propose(root, "notes/a.md", "x\n", write_allow=["notes/**"])
    store.save(proposal)
    # оператор перезапустил сервер без расширенного allowlist — запись недопустима
    assert code_of(lambda: fr.apply_proposal(root, store, proposal.id)) == fr.CODE_NOT_WRITABLE
    assert not (root / "notes").exists()


def test_apply_rejects_tampered_staging_path(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    proposal = propose(root, "README.md", "# Проект\n\nновое\n")
    store.save(proposal)
    staged = tmp_path / "staging" / f"{proposal.id}.json"
    data = json.loads(staged.read_text(encoding="utf-8"))
    data["path"] = "src/main.py"
    staged.write_text(json.dumps(data), encoding="utf-8")
    assert code_of(lambda: fr.apply_proposal(root, store, proposal.id)) == fr.CODE_NOT_WRITABLE
    assert (root / "src" / "main.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_apply_preserves_file_mode(tmp_path):
    root = make_root(tmp_path)
    store = make_store(tmp_path)
    os.chmod(root / "README.md", 0o600)
    proposal = propose(root, "README.md", "# Проект\n\nновое\n")
    store.save(proposal)
    fr.apply_proposal(root, store, proposal.id)
    assert (root / "README.md").stat().st_mode & 0o777 == 0o600


# --- защита от номеров строк в новом содержимом -------------------------------------------------


def test_numbered_new_content_rejected(tmp_path):
    root = make_root(tmp_path)
    numbered = "1: # Проект\n2: \n3: новое\n4: ещё\n"
    assert code_of(lambda: propose(root, "README.md", numbered)) == fr.CODE_BAD_ARGUMENT


def test_numbered_content_allowed_when_original_is_numbered(tmp_path):
    root = make_root(tmp_path)
    (root / "docs" / "steps.md").write_text("1: один\n2: два\n3: три\n", encoding="utf-8")
    proposal = propose(root, "docs/steps.md", "1: один\n2: два\n3: три\n4: четыре\n")
    assert proposal.added == 1


def test_looks_line_numbered_boundaries():
    assert not fr.looks_line_numbered("1: a\n2: b\n", None)  # меньше трёх строк
    assert not fr.looks_line_numbered("1: a\nпросто\nещё\nи ещё\n", None)  # доля < 0.8
    assert fr.looks_line_numbered("1: a\n2: b\n3: c\n", None)


# --- завершающий перевод строки ---------------------------------------------------------------


def test_missing_trailing_newline_restored_for_existing_file(tmp_path):
    root = make_root(tmp_path)
    proposal = propose(root, "README.md", "# Проект\n\nновое")
    assert proposal.new_content.endswith("новое\n")
    assert "No newline" not in proposal.diff


def test_existing_file_without_trailing_newline_keeps_it_that_way(tmp_path):
    root = make_root(tmp_path)
    (root / "docs" / "raw.md").write_text("a\nb", encoding="utf-8")
    proposal = propose(root, "docs/raw.md", "a\nc")
    assert proposal.new_content == "a\nc"


def test_new_file_gets_trailing_newline(tmp_path):
    root = make_root(tmp_path)
    assert propose(root, "docs/new.md", "# Новый").new_content == "# Новый\n"


def test_only_newline_difference_is_no_change(tmp_path):
    root = make_root(tmp_path)
    same = (root / "README.md").read_text(encoding="utf-8").rstrip("\n")
    assert code_of(lambda: propose(root, "README.md", same)) == fr.CODE_NO_CHANGE
