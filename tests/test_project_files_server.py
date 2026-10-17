"""Тесты обработчиков сервера файлов (mcp_servers/project_files/server.py): `FilesService` без
процесса и сети, на временном каталоге; сборка параметров запуска."""

from dataclasses import replace

import files_rules as fr
import pytest
import server as files_server


def make_service(tmp_path, **limit_overrides):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "data").mkdir()
    (root / "README.md").write_text("# Проект\n\nUSE_ME\n", encoding="utf-8")
    (root / "src" / "main.py").write_text("USE_ME = 1\n", encoding="utf-8")
    (root / ".env").write_text("USE_ME=secret\n", encoding="utf-8")
    limits = replace(fr.Limits(), **limit_overrides)
    return files_server.FilesService(root, tmp_path / "staging", limits), root


def test_list_read_search(tmp_path):
    service, _ = make_service(tmp_path)
    assert "README.md" in service.list_dir()["entries"]
    read = service.read_file("README.md", 3, 3)
    assert read["content"] == "3: USE_ME" and read["path"] == "README.md"
    found = service.search_text("USE_ME")
    assert {m["path"] for m in found["matches"]} == {"README.md", "src/main.py"}


def test_denied_everywhere(tmp_path):
    service, _ = make_service(tmp_path)
    with pytest.raises(fr.FilesError):
        service.read_file(".env")
    with pytest.raises(fr.FilesError):
        service.list_dir("data")
    assert ".env" not in service.list_dir()["entries"]


def test_read_directory_as_file_rejected(tmp_path):
    service, _ = make_service(tmp_path)
    with pytest.raises(fr.FilesError):
        service.read_file("")


def test_propose_does_not_change_file_and_apply_writes_atomically(tmp_path):
    service, root = make_service(tmp_path)
    before = (root / "README.md").read_text(encoding="utf-8")
    proposed = service.propose_change("README.md", "# Проект\n\nUSE_ME\nещё\n", "добавил строку")
    assert (root / "README.md").read_text(encoding="utf-8") == before
    assert proposed["added"] == 1 and proposed["removed"] == 0 and "ещё" in proposed["diff"]
    applied = service.apply_change(proposed["proposal_id"])
    assert applied["path"] == "README.md" and applied["diff"] == proposed["diff"]
    assert (root / "README.md").read_text(encoding="utf-8").endswith("ещё\n")
    with pytest.raises(fr.FilesError) as info:
        service.apply_change(proposed["proposal_id"])
    assert info.value.code == fr.CODE_UNKNOWN_PROPOSAL


def test_apply_rejects_changed_file(tmp_path):
    service, root = make_service(tmp_path)
    proposed = service.propose_change("README.md", "# Другое\n")
    (root / "README.md").write_text("правка руками\n", encoding="utf-8")
    with pytest.raises(fr.FilesError) as info:
        service.apply_change(proposed["proposal_id"])
    assert info.value.code == fr.CODE_CHANGED
    assert (root / "README.md").read_text(encoding="utf-8") == "правка руками\n"


def test_propose_source_code_rejected(tmp_path):
    service, _ = make_service(tmp_path)
    with pytest.raises(fr.FilesError) as info:
        service.propose_change("src/main.py", "x\n")
    assert info.value.code == fr.CODE_NOT_WRITABLE


def test_discard(tmp_path):
    service, root = make_service(tmp_path)
    proposed = service.propose_change("README.md", "# Другое\n")
    assert service.discard_change(proposed["proposal_id"]) == {"discarded": True}
    with pytest.raises(fr.FilesError):
        service.apply_change(proposed["proposal_id"])
    assert (root / "README.md").read_text(encoding="utf-8").startswith("# Проект")


def test_proposals_per_session_limit(tmp_path):
    service, _ = make_service(tmp_path, max_proposals=2)
    service.propose_change("README.md", "a\n")
    service.propose_change("docs/a.md", "b\n")
    with pytest.raises(fr.FilesError) as info:
        service.propose_change("docs/b.md", "c\n")
    assert info.value.code == fr.CODE_LIMIT


def test_write_allow_extension_by_operator(tmp_path):
    service, root = make_service(tmp_path)
    with pytest.raises(fr.FilesError):
        service.propose_change("notes/a.md", "x\n")
    extended = files_server.FilesService(
        root, tmp_path / "staging2", fr.Limits(), ("notes/**",)
    )
    assert extended.propose_change("notes/a.md", "x\n")["creates"] is True


def test_args_parse_and_build_service(tmp_path):
    args = files_server.parse_args(
        ["--root", str(tmp_path), "--staging-dir", str(tmp_path / "s"), "--max-proposals", "1",
         "--write-allow", "notes/**"]
    )
    service = files_server.service_from_args(args)
    assert service.limits.max_proposals == 1 and service.write_allow == ("notes/**",)


def test_tool_registration_has_no_root_parameters(tmp_path):
    service, _ = make_service(tmp_path)
    server = files_server.build_server(service)
    assert server is not None
