"""Чистые правила MCP-сервера файлов проекта (спека project-files-server; design.md изменения
add-dev-project-files, решения 2–4, 8): канонизация путей, denylist чтения, allowlist записи,
лимиты, diff, хэш и хранилище предложений.

Не импортирует код бота и MCP и не обращается к сети: корень, каталог предложений и пределы
передаются аргументами, поэтому правила проверяются тестами на временном каталоге
(tests/test_project_files_rules.py). Все границы доступа — здесь, на стороне сервера."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# --- ошибки ----------------------------------------------------------------------------------


class FilesError(Exception):
    """Отказ сервера с понятным текстом (его видит модель или бот). Текст не раскрывает
    содержимое и само существование запрещённых путей."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


CODE_OUTSIDE = "path_outside_root"
CODE_DENIED = "path_denied"
CODE_NOT_FOUND = "not_found"
CODE_NOT_FILE = "not_a_file"
CODE_NOT_DIR = "not_a_directory"
CODE_BINARY = "binary_file"
CODE_TOO_LARGE = "too_large"
CODE_NOT_WRITABLE = "path_not_writable"
CODE_NO_CHANGE = "no_change"
CODE_CRLF = "crlf_not_supported"
CODE_UNKNOWN_PROPOSAL = "unknown_proposal"
CODE_EXPIRED = "proposal_expired"
CODE_CHANGED = "source_changed"
CODE_EXISTS = "file_exists"
CODE_LIMIT = "limit_reached"
CODE_BAD_ARGUMENT = "bad_argument"

DENIED_MESSAGE = "Доступ к этому пути закрыт правилами сервера."
OUTSIDE_MESSAGE = "Путь выходит за пределы корня проекта."

# --- доступ ----------------------------------------------------------------------------------

# Каталоги, закрытые на любой глубине.
DENIED_DIR_NAMES = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "node_modules", "venv"}
)
# Каталоги, закрытые только в корне проекта.
DENIED_TOP_LEVEL = frozenset({"data"})
DENIED_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db")
DENIED_NAME_PREFIXES = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")
ENV_EXAMPLE = ".env.example"

# Запись разрешена только в документацию и образец окружения (решение 4). Расширение —
# аргументом запуска оператора (`--write-allow`), модель и пользователь чата его не меняют.
DEFAULT_WRITE_ALLOW = (
    "README.md",
    "CLAUDE.md",
    "src/**/CLAUDE.md",
    "docs/**",
    ".env.example",
    "CHANGELOG.md",
)


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Glob с `**` (любая глубина), `*` (в пределах сегмента) и `?` → регулярное выражение."""
    result = ["^"]
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if pattern.startswith("**/", i):
            result.append("(?:.*/)?")
            i += 3
            continue
        if pattern.startswith("**", i):
            result.append(".*")
            i += 2
            continue
        if char == "*":
            result.append("[^/]*")
        elif char == "?":
            result.append("[^/]")
        else:
            result.append(re.escape(char))
        i += 1
    result.append("$")
    return re.compile("".join(result))


def matches_glob(rel_path: str, pattern: str) -> bool:
    return _glob_to_regex(pattern).match(rel_path) is not None


def is_read_denied(rel_path: str) -> bool:
    """Путь (относительный, `/`) закрыт для чтения, поиска и списков: файлы окружения (кроме
    `.env.example`), `data/`, `.git/`, виртуальные окружения, кэши, ключи и базы."""
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return False
    if parts[0] in DENIED_TOP_LEVEL:
        return True
    for part in parts[:-1]:
        if part in DENIED_DIR_NAMES or part.startswith(".venv"):
            return True
    last = parts[-1]
    if last in DENIED_DIR_NAMES or last.startswith(".venv"):
        return True
    for part in parts:
        if part.startswith(".env") and part != ENV_EXAMPLE:
            return True
    lowered = last.lower()
    if lowered.endswith(DENIED_SUFFIXES) or ".sqlite" in lowered:
        return True
    return last.startswith(DENIED_NAME_PREFIXES)


def is_write_allowed(rel_path: str, extra_globs: Iterable[str] = ()) -> bool:
    """Запись разрешена: путь не закрыт для чтения И подходит под allowlist записи."""
    if not rel_path or is_read_denied(rel_path):
        return False
    return any(matches_glob(rel_path, p) for p in (*DEFAULT_WRITE_ALLOW, *extra_globs))


def resolve_path(root: Path, raw: str) -> tuple[Path, str]:
    """Канонизация пути: `(абсолютный канонический путь, относительный posix-путь)`.

    Разрешает `..`, абсолютные пути и символические ссылки (`Path.resolve`), проверяет, что
    итог внутри корня, и применяет denylist к КАНОНИЧЕСКОМУ относительному пути — поэтому
    ссылка внутри корня на `.env` отклоняется. Пустой путь и `.` — сам корень (`rel == ""`)."""
    if not isinstance(raw, str) or "\x00" in raw:
        raise FilesError(CODE_BAD_ARGUMENT, "Путь должен быть строкой без служебных символов.")
    root_real = root.resolve()
    candidate = Path(raw.strip() or ".")
    absolute = candidate if candidate.is_absolute() else root_real / candidate
    try:
        canonical = absolute.resolve()
    except (OSError, RuntimeError):
        raise FilesError(CODE_BAD_ARGUMENT, "Путь не удалось разобрать.") from None
    if not canonical.is_relative_to(root_real):
        raise FilesError(CODE_OUTSIDE, OUTSIDE_MESSAGE)
    rel = canonical.relative_to(root_real).as_posix()
    rel = "" if rel == "." else rel
    if is_read_denied(rel):
        raise FilesError(CODE_DENIED, DENIED_MESSAGE)
    return canonical, rel


# --- лимиты ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Limits:
    """Пределы сервера (аргументы запуска оператора)."""

    max_read_chars: int = 30000
    max_file_bytes: int = 1_000_000
    max_search_results: int = 50
    max_response_chars: int = 40000
    max_proposal_chars: int = 40000
    max_proposals: int = 3
    ttl_seconds: int = 3600
    context_lines: int = 3
    max_line_chars: int = 300


TRUNCATED_NOTE = "[усечено]"


def clip(text: str, limit: int) -> tuple[str, bool]:
    """Обрезка до `limit` символов: `(текст, усечён ли)`."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


# --- чтение ----------------------------------------------------------------------------------


def is_binary(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def read_text(path: Path, limits: Limits) -> str:
    """Текст файла. Двоичный, слишком большой, не файл — отказ с причиной (не частичное чтение)."""
    if not path.exists():
        raise FilesError(CODE_NOT_FOUND, "Файл не найден.")
    if not path.is_file():
        raise FilesError(CODE_NOT_FILE, "Это не файл.")
    size = path.stat().st_size
    if size > limits.max_file_bytes:
        raise FilesError(
            CODE_TOO_LARGE,
            f"Файл слишком большой ({size} байт, предел {limits.max_file_bytes}); "
            "читай его частями через поиск.",
        )
    data = path.read_bytes()
    if is_binary(data):
        raise FilesError(CODE_BINARY, "Файл двоичный или не в UTF-8: читать его нельзя.")
    return data.decode("utf-8")


def read_numbered(
    text: str, start_line: int | None, end_line: int | None, limits: Limits
) -> dict:
    """Строки с номерами (`N: текст`) в пределах диапазона и `max_read_chars`; усечение явное."""
    lines = text.splitlines()
    total = len(lines)
    start = 1 if start_line is None else start_line
    end = total if end_line is None else end_line
    if start < 1 or end < start:
        raise FilesError(CODE_BAD_ARGUMENT, "Диапазон строк задан неверно (с 1, конец ≥ начала).")
    end = min(end, total)
    out: list[str] = []
    used = 0
    last = start - 1
    truncated = False
    for number in range(start, end + 1):
        line = f"{number}: {lines[number - 1]}"
        if used + len(line) + 1 > limits.max_read_chars:
            truncated = True
            break
        out.append(line)
        used += len(line) + 1
        last = number
    if not out and start <= end:
        truncated = True
    return {
        "start_line": start,
        "end_line": last,
        "total_lines": total,
        "truncated": truncated,
        "content": "\n".join(out),
    }


def list_directory(directory: Path, root: Path, limits: Limits) -> dict:
    """Содержимое каталога без запрещённых путей; каталоги — с `/` на конце."""
    if not directory.exists():
        raise FilesError(CODE_NOT_FOUND, "Каталог не найден.")
    if not directory.is_dir():
        raise FilesError(CODE_NOT_DIR, "Это не каталог.")
    root_real = root.resolve()
    entries: list[str] = []
    for child in sorted(directory.iterdir(), key=lambda p: p.name):
        try:
            rel = child.resolve().relative_to(root_real).as_posix()
        except (ValueError, OSError):
            continue  # ссылка наружу: в списке её нет
        if is_read_denied(rel):
            continue
        entries.append(child.name + ("/" if child.is_dir() else ""))
    entries, truncated = _limit_items(entries, limits.max_search_results * 10)
    return {"entries": entries, "truncated": truncated}


def _limit_items(items: list, limit: int) -> tuple[list, bool]:
    return (items[:limit], True) if len(items) > limit else (items, False)


def iter_files(root: Path) -> Iterable[tuple[Path, str]]:
    """Обход корня: `(путь, относительный posix-путь)` только разрешённых для чтения файлов.
    Запретные каталоги не обходятся; ссылки наружу и на запретное пропускаются."""
    root_real = root.resolve()
    for current, dirs, files in os.walk(root_real, followlinks=False):
        current_path = Path(current)
        rel_dir = current_path.relative_to(root_real).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirs[:] = sorted(
            d for d in dirs if not is_read_denied(f"{rel_dir}/{d}" if rel_dir else d)
        )
        for name in sorted(files):
            path = current_path / name
            try:
                canonical = path.resolve()
                rel = canonical.relative_to(root_real).as_posix()
            except (ValueError, OSError):
                continue
            if is_read_denied(rel) or not canonical.is_file():
                continue
            yield canonical, rel


def search(
    root: Path,
    query: str,
    limits: Limits,
    path_glob: str | None = None,
    regex: bool = False,
) -> dict:
    """Поиск текста по файлам проекта: совпадения `{path, line, text}`, не более
    `max_search_results`; двоичные и слишком большие файлы пропускаются; усечение явное."""
    if not isinstance(query, str) or not query:
        raise FilesError(CODE_BAD_ARGUMENT, "Строка поиска не может быть пустой.")
    if regex:
        try:
            pattern = re.compile(query)
        except re.error as exc:
            raise FilesError(
                CODE_BAD_ARGUMENT, f"Регулярное выражение некорректно: {exc}."
            ) from None

        def hit(line: str) -> bool:
            return pattern.search(line) is not None
    else:

        def hit(line: str) -> bool:
            return query in line

    matches: list[dict] = []
    skipped = 0
    truncated = False
    for path, rel in iter_files(root):
        if path_glob and not matches_glob(rel, path_glob):
            continue
        try:
            if path.stat().st_size > limits.max_file_bytes:
                skipped += 1
                continue
            data = path.read_bytes()
        except OSError:
            skipped += 1
            continue
        if is_binary(data):
            skipped += 1
            continue
        for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            if hit(line):
                if len(matches) >= limits.max_search_results:
                    truncated = True
                    break
                text, _ = clip(line.strip(), limits.max_line_chars)
                matches.append({"path": rel, "line": number, "text": text})
        if truncated:
            break
    return {"matches": matches, "count": len(matches), "truncated": truncated, "skipped": skipped}


# --- diff и хэш ------------------------------------------------------------------------------


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def build_diff(rel_path: str, old: str | None, new: str, context: int = 3) -> str:
    """Unified diff в формате `git apply`. `old=None` — создание файла. Детерминирован: один и
    тот же вход даёт один и тот же текст (фиксированный контекст, окончания строк — LF)."""
    old_lines = [] if old is None else _lines(old)
    new_lines = _lines(new)
    body = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="/dev/null" if old is None else f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=context,
        )
    )
    if not body:
        return ""
    out = [f"diff --git a/{rel_path} b/{rel_path}\n"]
    if old is None:
        out.append("new file mode 100644\n")
    for line in body:
        if line.endswith("\n"):
            out.append(line)
        else:
            out.append(line + "\n\\ No newline at end of file\n")
    return "".join(out)


def count_changes(diff: str) -> tuple[int, int]:
    """`(добавлено, удалено)` строк по тексту diff (заголовки не считаются)."""
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "diff --git")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


# --- предложения -----------------------------------------------------------------------------

PROPOSAL_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+: ")
NUMBERED_CONTENT_MESSAGE = (
    "Новое содержимое похоже на вывод read_file с номерами строк («N: текст»). Передай файл "
    "целиком БЕЗ номеров строк и без префиксов «N: »."
)


def looks_line_numbered(new_text: str, old_text: str | None) -> bool:
    """Модель скопировала формат `read_file` (`N: текст`) в новое содержимое: почти все непустые
    строки начинаются с номера, а в исходном файле так не было. Такую правку испортила бы файл."""
    lines = [line for line in new_text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    def share(items: list[str]) -> float:
        return sum(1 for item in items if _NUMBERED_LINE_RE.match(item)) / len(items)

    if share(lines) < 0.8:
        return False
    old_lines = [line for line in (old_text or "").splitlines() if line.strip()]
    return not old_lines or share(old_lines) < 0.5


@dataclass(frozen=True)
class Proposal:
    id: str
    path: str
    creates: bool
    base_hash: str | None
    new_content: str
    summary: str
    created_at: float
    diff: str
    added: int
    removed: int

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "creates": self.creates,
            "base_hash": self.base_hash,
            "new_content": self.new_content,
            "summary": self.summary,
            "created_at": self.created_at,
            "diff": self.diff,
            "added": self.added,
            "removed": self.removed,
        }

    @staticmethod
    def from_json(data: dict) -> Proposal:
        return Proposal(
            id=data["id"],
            path=data["path"],
            creates=bool(data["creates"]),
            base_hash=data["base_hash"],
            new_content=data["new_content"],
            summary=data.get("summary", ""),
            created_at=float(data["created_at"]),
            diff=data["diff"],
            added=int(data["added"]),
            removed=int(data["removed"]),
        )


def prepare_proposal(
    root: Path,
    raw_path: str,
    new_content: str,
    summary: str,
    limits: Limits,
    write_allow: Iterable[str] = (),
    *,
    proposal_id: str | None = None,
    now: float | None = None,
) -> Proposal:
    """Предложение правки: проверки пути (доступ + allowlist записи), размера, «без изменений»,
    diff и хэш исходника. Файлы проекта НЕ изменяются."""
    if not isinstance(new_content, str):
        raise FilesError(CODE_BAD_ARGUMENT, "Новое содержимое должно быть строкой.")
    if len(new_content) > limits.max_proposal_chars:
        raise FilesError(
            CODE_TOO_LARGE,
            f"Новое содержимое слишком большое ({len(new_content)} символов, предел "
            f"{limits.max_proposal_chars}).",
        )
    canonical, rel = resolve_path(root, raw_path)
    if not rel or not is_write_allowed(rel, write_allow):
        raise FilesError(
            CODE_NOT_WRITABLE,
            "В этот путь запись недопустима: правки готовятся только для документации "
            "(README.md, CLAUDE.md, docs/, .env.example, CHANGELOG.md). Опиши правку текстом.",
        )
    new_text = new_content.replace("\r\n", "\n")
    if canonical.exists():
        if not canonical.is_file():
            raise FilesError(CODE_NOT_FILE, "Это не файл.")
        raw = canonical.read_bytes()
        if is_binary(raw):
            raise FilesError(CODE_BINARY, "Файл двоичный или не в UTF-8.")
        if b"\r\n" in raw:
            raise FilesError(
                CODE_CRLF, "В файле окончания строк CRLF: такие файлы сервер не правит."
            )
        old_text: str | None = raw.decode("utf-8")
        base_hash: str | None = content_hash(raw)
    else:
        old_text = None
        base_hash = None
    # Модели часто теряют завершающий перевод строки: у существующего файла он сохраняется как был,
    # новый файл заканчивается переводом строки.
    if new_text and not new_text.endswith("\n") and (old_text is None or old_text.endswith("\n")):
        new_text += "\n"
    if old_text is not None and old_text == new_text:
        raise FilesError(CODE_NO_CHANGE, "Изменений нет: новое содержимое равно текущему.")
    if looks_line_numbered(new_text, old_text):
        raise FilesError(CODE_BAD_ARGUMENT, NUMBERED_CONTENT_MESSAGE)
    diff = build_diff(rel, old_text, new_text, limits.context_lines)
    if not diff:
        raise FilesError(CODE_NO_CHANGE, "Изменений нет: новое содержимое равно текущему.")
    added, removed = count_changes(diff)
    return Proposal(
        id=proposal_id or uuid.uuid4().hex[:12],
        path=rel,
        creates=old_text is None,
        base_hash=base_hash,
        new_content=new_text,
        summary=clip(summary.strip(), 300)[0] if isinstance(summary, str) else "",
        created_at=time.time() if now is None else now,
        diff=diff,
        added=added,
        removed=removed,
    )


class ProposalStore:
    """Предложения на диске (`<id>.json` в `staging_dir`): состояние переживает процесс сервера,
    живущий один вопрос. Запись атомарна; просроченные удаляются при обращении; `take` вынимает
    предложение (переименование) ДО любой записи в проект — второй вызов получает «неизвестное»."""

    def __init__(
        self, staging_dir: Path, ttl_seconds: int, clock: Callable[[], float] = time.time
    ) -> None:
        self._dir = staging_dir
        self._ttl = ttl_seconds
        self._clock = clock

    def _file(self, proposal_id: str) -> Path:
        if not PROPOSAL_ID_RE.match(proposal_id or ""):
            raise FilesError(CODE_UNKNOWN_PROPOSAL, "Неизвестное предложение.")
        return self._dir / f"{proposal_id}.json"

    def save(self, proposal: Proposal) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self.purge_expired()
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(proposal.to_json(), handle, ensure_ascii=False)
            os.replace(tmp, self._file(proposal.id))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _is_expired(self, created_at: float) -> bool:
        return self._clock() - created_at > self._ttl

    def purge_expired(self) -> int:
        """Удаляет просроченные (и повреждённые) предложения; возвращает число удалённых."""
        removed = 0
        if not self._dir.is_dir():
            return 0
        for path in self._dir.glob("*.json"):
            try:
                created = float(json.loads(path.read_text(encoding="utf-8"))["created_at"])
            except (OSError, ValueError, KeyError, TypeError):
                path.unlink(missing_ok=True)
                removed += 1
                continue
            if self._is_expired(created):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def take(self, proposal_id: str) -> Proposal:
        """Вынимает предложение для применения. Отсутствует/уже взято → «неизвестное»;
        просрочено → удаляется и «просрочено»."""
        path = self._file(proposal_id)
        taken = path.with_name(f"{proposal_id}.{uuid.uuid4().hex[:6]}.taken")
        try:
            os.rename(path, taken)
        except FileNotFoundError:
            raise FilesError(CODE_UNKNOWN_PROPOSAL, "Неизвестное предложение.") from None
        try:
            proposal = Proposal.from_json(json.loads(taken.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            raise FilesError(CODE_UNKNOWN_PROPOSAL, "Предложение повреждено.") from None
        finally:
            taken.unlink(missing_ok=True)
        if self._is_expired(proposal.created_at):
            raise FilesError(CODE_EXPIRED, "Срок жизни предложения истёк; подготовь его заново.")
        return proposal

    def discard(self, proposal_id: str) -> bool:
        path = self._file(proposal_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def count(self) -> int:
        return len(list(self._dir.glob("*.json"))) if self._dir.is_dir() else 0


def write_atomic(path: Path, text: str) -> None:
    """Атомарная запись: временный файл рядом + `os.replace`; права существующего файла
    сохраняются."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o7777)
        else:
            os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def apply_proposal(
    root: Path,
    store: ProposalStore,
    proposal_id: str,
    write_allow: Iterable[str] = (),
) -> Proposal:
    """Применение: предложение вынимается из хранилища ДО записи; путь заново проверяется по
    правилам доступа и записи; хэш текущего файла сверяется с зафиксированным; запись атомарна."""
    proposal = store.take(proposal_id)
    canonical, rel = resolve_path(root, proposal.path)
    if rel != proposal.path or not is_write_allowed(rel, write_allow):
        raise FilesError(CODE_NOT_WRITABLE, "В этот путь запись недопустима.")
    if proposal.creates:
        if canonical.exists():
            raise FilesError(
                CODE_EXISTS, "Файл уже создан другим способом: исходное содержимое изменилось."
            )
    else:
        if not canonical.is_file():
            raise FilesError(CODE_CHANGED, "Файл исчез: исходное содержимое изменилось.")
        if content_hash(canonical.read_bytes()) != proposal.base_hash:
            raise FilesError(CODE_CHANGED, "Исходное содержимое файла изменилось после подготовки.")
    write_atomic(canonical, proposal.new_content)
    return proposal
