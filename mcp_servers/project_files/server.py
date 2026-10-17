"""MCP-сервер файлов проекта для режима `/dev` (спека project-files-server; design.md изменения
add-dev-project-files, решения 1–5): список каталога, чтение и поиск (только чтение), подготовка
предложения правки (в проект не пишет) и — отдельно, для кода бота — применение и отклонение.

Все границы доступа здесь: корень и каталог предложений задаёт ТОЛЬКО оператор аргументами
запуска (в схемах инструментов их нет), denylist чтения и allowlist записи — `files_rules.py`.
Не часть бота: `src/` этот модуль не импортирует, сеть не используется.

Запуск: `python mcp_servers/project_files/server.py --root <репозиторий> --staging-dir <каталог>`"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import files_rules as fr
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(read_only_hint=True)
WRITES_STAGING = ToolAnnotations(read_only_hint=False, destructive_hint=False)
WRITES_PROJECT = ToolAnnotations(read_only_hint=False, destructive_hint=True)

SESSION_LIMIT_MESSAGE = (
    "Лимит предложений на один вопрос исчерпан ({limit}): остальные правки опиши текстом."
)


class FilesService:
    """Обработчики инструментов без MCP: тестируются без процесса. Каждый возвращает словарь
    либо поднимает `FilesError` (текст готов для модели)."""

    def __init__(
        self,
        root: Path,
        staging_dir: Path,
        limits: fr.Limits,
        write_allow: tuple[str, ...] = (),
        store: fr.ProposalStore | None = None,
    ) -> None:
        self.root = root.resolve()
        self.limits = limits
        self.write_allow = tuple(write_allow)
        self.store = store or fr.ProposalStore(staging_dir, limits.ttl_seconds)
        self._proposed = 0  # предложений в этом процессе (= одном вопросе)

    def list_dir(self, path: str = "") -> dict[str, Any]:
        canonical, rel = fr.resolve_path(self.root, path)
        result = fr.list_directory(canonical, self.root, self.limits)
        return {"path": rel or ".", **result}

    def read_file(
        self, path: str, start_line: int | None = None, end_line: int | None = None
    ) -> dict[str, Any]:
        canonical, rel = fr.resolve_path(self.root, path)
        if not rel:
            raise fr.FilesError(fr.CODE_NOT_FILE, "Это каталог, а не файл.")
        text = fr.read_text(canonical, self.limits)
        return {"path": rel, **fr.read_numbered(text, start_line, end_line, self.limits)}

    def search_text(
        self, query: str, path_glob: str | None = None, regex: bool = False
    ) -> dict[str, Any]:
        return fr.search(self.root, query, self.limits, path_glob, regex)

    def propose_change(self, path: str, new_content: str, summary: str = "") -> dict[str, Any]:
        if self._proposed >= self.limits.max_proposals:
            raise fr.FilesError(
                fr.CODE_LIMIT, SESSION_LIMIT_MESSAGE.format(limit=self.limits.max_proposals)
            )
        proposal = fr.prepare_proposal(
            self.root, path, new_content, summary, self.limits, self.write_allow
        )
        self.store.save(proposal)
        self._proposed += 1
        return {
            "proposal_id": proposal.id,
            "path": proposal.path,
            "creates": proposal.creates,
            "added": proposal.added,
            "removed": proposal.removed,
            "summary": proposal.summary,
            "base_hash": proposal.base_hash,
            "diff": proposal.diff,
        }

    def apply_change(self, proposal_id: str) -> dict[str, Any]:
        proposal = fr.apply_proposal(self.root, self.store, proposal_id, self.write_allow)
        return {
            "path": proposal.path,
            "creates": proposal.creates,
            "added": proposal.added,
            "removed": proposal.removed,
            "diff": proposal.diff,
        }

    def discard_change(self, proposal_id: str) -> dict[str, Any]:
        if not self.store.discard(proposal_id):
            raise fr.FilesError(fr.CODE_UNKNOWN_PROPOSAL, "Неизвестное предложение.")
        return {"discarded": True}


def build_server(service: FilesService) -> MCPServer:
    """Регистрирует инструменты. `FilesError` превращается в `ToolError`: результат помечен
    `is_error`, а текст доходит до модели (любое иное исключение SDK скрывает)."""
    server = MCPServer("project-files")

    def guarded(call):
        try:
            return call()
        except fr.FilesError as exc:
            raise ToolError(exc.message) from None

    @server.tool(annotations=READ_ONLY)
    def list_dir(path: str = "") -> dict[str, Any]:
        """Список файлов и каталогов проекта по относительному пути (пусто — корень проекта).
        Каталоги помечены `/` на конце; закрытые пути (секреты, data, .git) не показываются."""
        return guarded(lambda: service.list_dir(path))

    @server.tool(annotations=READ_ONLY)
    def read_file(
        path: str, start_line: int | None = None, end_line: int | None = None
    ) -> dict[str, Any]:
        """Текст файла проекта с номерами строк (`N: текст`). start_line/end_line — необязательный
        диапазон (с 1). Большие, двоичные и закрытые файлы читать нельзя; усечение отмечено."""
        return guarded(lambda: service.read_file(path, start_line, end_line))

    @server.tool(annotations=READ_ONLY)
    def search_text(
        query: str, path_glob: str | None = None, regex: bool = False
    ) -> dict[str, Any]:
        """Поиск строки (или регулярного выражения при regex=true) по файлам проекта. Результат —
        совпадения `path`, `line`, `text`; path_glob (например `docs/**`) сужает область."""
        return guarded(lambda: service.search_text(query, path_glob, regex))

    @server.tool(annotations=WRITES_STAGING)
    def propose_change(path: str, new_content: str, summary: str = "") -> dict[str, Any]:
        """Подготовить предложение правки файла документации: path и ПОЛНОЕ новое содержимое файла.
        new_content — чистый текст файла БЕЗ номеров строк (формат «N: » у read_file не копируй).
        Файл не меняется: сервер строит diff, пользователь видит его и сам нажимает «Применить».
        Запись разрешена только в README.md, CLAUDE.md, docs/, .env.example, CHANGELOG.md."""
        return guarded(lambda: service.propose_change(path, new_content, summary))

    @server.tool(annotations=WRITES_PROJECT)
    def apply_change(proposal_id: str) -> dict[str, Any]:
        """Применить предложение по идентификатору (только для кода бота, модели не отдаётся)."""
        return guarded(lambda: service.apply_change(proposal_id))

    @server.tool(annotations=WRITES_STAGING)
    def discard_change(proposal_id: str) -> dict[str, Any]:
        """Отклонить предложение без записи (только для кода бота, модели не отдаётся)."""
        return guarded(lambda: service.discard_change(proposal_id))

    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = fr.Limits()
    parser = argparse.ArgumentParser(description="MCP-сервер файлов проекта (режим /dev).")
    parser.add_argument("--root", required=True, help="Корень проекта (доступ только внутри)")
    parser.add_argument("--staging-dir", required=True, help="Каталог ожидающих предложений")
    parser.add_argument(
        "--write-allow", action="append", default=[], help="Дополнительный glob путей для записи"
    )
    parser.add_argument("--ttl-seconds", type=int, default=defaults.ttl_seconds)
    parser.add_argument("--max-read-chars", type=int, default=defaults.max_read_chars)
    parser.add_argument("--max-search-results", type=int, default=defaults.max_search_results)
    parser.add_argument("--max-proposal-chars", type=int, default=defaults.max_proposal_chars)
    parser.add_argument("--max-proposals", type=int, default=defaults.max_proposals)
    return parser.parse_args(argv)


def service_from_args(args: argparse.Namespace) -> FilesService:
    limits = fr.Limits(
        max_read_chars=args.max_read_chars,
        max_search_results=args.max_search_results,
        max_proposal_chars=args.max_proposal_chars,
        max_proposals=args.max_proposals,
        ttl_seconds=args.ttl_seconds,
    )
    return FilesService(
        Path(args.root), Path(args.staging_dir), limits, tuple(args.write_allow)
    )


def main() -> None:
    build_server(service_from_args(parse_args())).run("stdio")


if __name__ == "__main__":
    main()
