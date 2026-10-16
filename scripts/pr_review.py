"""AI-ревью pull request: оснастка вне бота (спека `openspec/changes/add-pr-ai-review/`).

    python scripts/pr_review.py index
    python scripts/pr_review.py review --pr 42 --base <sha|ветка> --head <sha> [--dry-run]

`index` собирает индекс ревью из рабочего каталога ОСНОВНОЙ ветки (документация + исходники
`src/`, `scripts/`) в `PR_REVIEW_INDEX_DIR`. `review` берёт diff и файлы PR командами
`git diff`/`git show` для заданных коммитов, ищет контекст в индексе, просит модель о ревью и
публикует (или обновляет) один комментарий в PR через `gh api`; `--dry-run` печатает комментарий
вместо публикации.

Модель угроз: код PR — только ДАННЫЕ. Скрипт не делает checkout PR, не импортирует и не исполняет
его файлы, не ставит зависимостей, не отдаёт модели инструментов; результат — только текст
комментария. Скрипт и workflow берутся из основной ветки (`pull_request_target`).

Настройки (окружение runner; задаёт ОПЕРАТОР, автор PR их не меняет):
    PR_REVIEW_INDEX_DIR        каталог индекса ревью (по умолчанию data/pr_review_index)
    PR_REVIEW_PROVIDER         подпись провайдера в комментарии (по умолчанию ollama)
    PR_REVIEW_MODEL            модель ревью — обязательна
    PR_REVIEW_BASE_URL         OpenAI-совместимый адрес (для ollama по умолчанию localhost:11434)
    PR_REVIEW_API_KEY          ключ облачного провайдера (для ollama не нужен)
    PR_REVIEW_TIMEOUT_SECONDS  таймаут ответа модели (по умолчанию 300)
    PR_REVIEW_MAX_CHARS        бюджет diff и файлов, символов (по умолчанию 24000)
    PR_REVIEW_MAX_TOKENS       предел ответа модели, включая рассуждение (по умолчанию 8192)
    PR_REVIEW_TOP_K            сколько фрагментов индекса подаётся модели (по умолчанию 8)
    EMBEDDINGS_BASE_URL / EMBEDDINGS_MODEL  эмбеддинги (по умолчанию localhost Ollama, bge-m3)
    GITHUB_REPOSITORY          owner/repo (задаёт GitHub Actions); GH_TOKEN — токен для `gh`

Код выхода: 0 — готово; 1 — ревью не выполнено или публикация не удалась; 2 — ошибка настроек.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pr_review_index as index_lib  # noqa: E402, I001
import pr_review_lib as lib  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_CHARS = 24000
DEFAULT_MAX_TOKENS = 8192
# Сверх этого размера файл PR не читается целиком (ревью по diff), вывод `git diff` обрезается.
MAX_FILE_READ_CHARS = 400_000
MAX_DIFF_READ_CHARS = 4_000_000
GIT_TIMEOUT_SECONDS = 120.0
GH_TIMEOUT_SECONDS = 60.0
BOT_LOGIN = "github-actions[bot]"


class ConfigError(Exception):
    """Ошибка настроек: сообщение оператору, код выхода 2."""


# --------------------------------------------------------------------------- #
# Настройки
# --------------------------------------------------------------------------- #


def _env_number(name: str, default: float, *, integer: bool = False) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r}: ожидается число") from None
    if value <= 0:
        raise ConfigError(f"{name}={raw!r}: ожидается положительное число")
    return value


@dataclass(frozen=True)
class Settings:
    index_dir: str
    embeddings_base_url: str
    embeddings_model: str
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout: float
    max_chars: int
    max_tokens: int
    top_k: int


def load_settings(*, need_model: bool) -> Settings:
    provider = os.getenv("PR_REVIEW_PROVIDER", "").strip() or "ollama"
    model = os.getenv("PR_REVIEW_MODEL", "").strip()
    base_url = os.getenv("PR_REVIEW_BASE_URL", "").strip()
    if need_model:
        if not model:
            raise ConfigError("PR_REVIEW_MODEL не задан: модель ревью выбирает оператор")
        if not base_url:
            if provider != "ollama":
                raise ConfigError(f"PR_REVIEW_BASE_URL обязателен для провайдера {provider!r}")
            base_url = DEFAULT_OLLAMA_URL
    index_dir = os.getenv("PR_REVIEW_INDEX_DIR", "").strip() or os.path.join(
        ROOT, "data", "pr_review_index"
    )
    return Settings(
        index_dir=index_dir,
        embeddings_base_url=os.getenv("EMBEDDINGS_BASE_URL", "").strip() or DEFAULT_OLLAMA_URL,
        embeddings_model=os.getenv("EMBEDDINGS_MODEL", "").strip() or "bge-m3",
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=os.getenv("PR_REVIEW_API_KEY", "").strip(),
        timeout=_env_number("PR_REVIEW_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        max_chars=int(_env_number("PR_REVIEW_MAX_CHARS", DEFAULT_MAX_CHARS, integer=True)),
        max_tokens=int(_env_number("PR_REVIEW_MAX_TOKENS", DEFAULT_MAX_TOKENS, integer=True)),
        top_k=int(_env_number("PR_REVIEW_TOP_K", lib.DEFAULT_TOP_K, integer=True)),
    )


# --------------------------------------------------------------------------- #
# git: только чтение, без checkout и без исполнения кода PR
# --------------------------------------------------------------------------- #


def run_git(args: list[str], *, cwd: str = ROOT) -> str:
    """Команда git без оболочки. `textconv` и внешние драйверы diff отключаются в самих командах
    (`--no-textconv`, `--no-ext-diff`): файлы PR не должны запускать настроенные фильтры."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]} завершился с кодом {result.returncode}")
    return result.stdout


def read_pr_diff(base: str, head: str) -> tuple[str, bool]:
    """Diff PR относительно общего предка (`base...head`); второй элемент — обрезан ли вывод."""
    out = run_git(
        [
            "-c", "core.quotepath=off", "diff", "--no-color", "--no-ext-diff", "--no-textconv",
            "-M", f"{base}...{head}",
        ]
    )  # fmt: skip
    if len(out) > MAX_DIFF_READ_CHARS:
        return out[:MAX_DIFF_READ_CHARS], True
    return out, False


def read_file_at(head: str, path: str) -> str | None:
    """Содержимое файла в коммите PR (`git show <sha>:<path>`) или None, если не прочитать."""
    try:
        text = run_git(["show", f"{head}:{path}"])
    except (RuntimeError, subprocess.TimeoutExpired):
        return None
    return text if len(text) <= MAX_FILE_READ_CHARS else None


# --------------------------------------------------------------------------- #
# Эмбеддинги и модель
# --------------------------------------------------------------------------- #


def make_embedder(base_url: str, model: str):
    """Тонкий клиент эмбеддингов (без импорта `config`): OpenAI-совместимый `/v1/embeddings`.
    Прокси из окружения не используются — Ollama локальный (как в providers/embeddings_client)."""
    import httpx
    from openai import OpenAI

    client = OpenAI(
        api_key="ollama",
        base_url=base_url,
        http_client=httpx.Client(trust_env=False),
        max_retries=3,
        timeout=60.0,
    )

    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda i: i.index)]

    return embed


def make_chat(settings: Settings):
    """Клиент модели ревью БЕЗ инструментов: на вход — сообщения, на выход — текст."""
    import httpx
    from openai import OpenAI

    kwargs = {}
    if settings.provider == "ollama":
        kwargs["http_client"] = httpx.Client(trust_env=False)
    client = OpenAI(
        api_key=settings.api_key or "ollama",
        base_url=settings.base_url,
        timeout=settings.timeout,
        max_retries=0,
        **kwargs,
    )

    def ask(messages: list[dict[str, str]]) -> tuple[str, str | None]:
        """Текст ответа и причина остановки (`length` — упёрся в лимит токенов)."""
        response = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=0.2,
            max_tokens=settings.max_tokens,
        )
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason

    return ask


class EmptyAnswerError(Exception):
    """Модель вернула пустой ответ (причина остановки — `finish`)."""

    def __init__(self, finish: str | None, max_tokens: int) -> None:
        super().__init__("пустой ответ модели")
        self.finish = finish
        self.max_tokens = max_tokens


def failure_reason(exc: Exception, settings: Settings) -> str:
    """Причина сбоя модели для комментария: без адресов, ключей и внутренних деталей."""
    import openai

    if isinstance(exc, EmptyAnswerError):
        if exc.finish == "length":
            return (
                f"ответ модели пуст: лимит {exc.max_tokens} токенов (PR_REVIEW_MAX_TOKENS) ушёл на "
                "рассуждение; оператору нужно увеличить лимит"
            )
        return "модель вернула пустой ответ"

    if isinstance(exc, openai.APITimeoutError):
        return f"сервер модели не ответил за {settings.timeout:g} с"
    if isinstance(exc, openai.APIConnectionError):
        return "сервер модели недоступен"
    if isinstance(exc, openai.AuthenticationError):
        return "провайдер модели отклонил ключ доступа"
    if isinstance(exc, openai.RateLimitError):
        return "провайдер модели ограничил число запросов"
    if isinstance(exc, openai.APIStatusError):
        return f"сервер модели вернул ошибку {exc.status_code}"
    return f"непредвиденная ошибка при обращении к модели ({type(exc).__name__})"


# --------------------------------------------------------------------------- #
# GitHub через gh
# --------------------------------------------------------------------------- #


def run_gh(args: list[str], *, stdin: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["gh", *args],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=GH_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("не найден `gh`") from None
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:2])} завершился с кодом {result.returncode}")
    return result.stdout


def fetch_pr_meta(repo: str, number: int) -> tuple[str, str]:
    data = json.loads(run_gh(["api", f"repos/{repo}/pulls/{number}"]))
    return str(data.get("title") or ""), str(data.get("body") or "")


def find_review_comment(repo: str, number: int, author: str = BOT_LOGIN) -> int | None:
    """id комментария ревью: тело начинается с маркера, автор — бот. Чужой комментарий с таким
    началом не трогается (иначе токен записи перезаписал бы чужой текст)."""
    jq = (
        f'.[] | select(.user.login == "{author}" and (.body | startswith("{lib.MARKER}"))) | .id'
    )
    out = run_gh(
        ["api", "--paginate", f"repos/{repo}/issues/{number}/comments", "--jq", jq]
    )
    ids = [line.strip() for line in out.splitlines() if line.strip().isdigit()]
    return int(ids[0]) if ids else None


def publish_comment(repo: str, number: int, body: str) -> str:
    """Обновляет комментарий ревью (PATCH) или создаёт его (POST); возвращает действие."""
    payload = json.dumps({"body": body}, ensure_ascii=False)
    existing = find_review_comment(repo, number, os.getenv("PR_REVIEW_COMMENT_AUTHOR", BOT_LOGIN))
    if existing is not None:
        run_gh(
            ["api", "--method", "PATCH", f"repos/{repo}/issues/comments/{existing}",
             "--input", "-"],
            stdin=payload,
        )  # fmt: skip
        return "обновлён"
    run_gh(
        ["api", "--method", "POST", f"repos/{repo}/issues/{number}/comments", "--input", "-"],
        stdin=payload,
    )
    return "создан"


# --------------------------------------------------------------------------- #
# Подкоманды
# --------------------------------------------------------------------------- #


def cmd_index(_args: argparse.Namespace) -> int:
    from rag.index_store import IndexRecord, build_index

    settings = load_settings(need_model=False)
    tracked = run_git(["ls-files"]).splitlines()
    files = lib.select_index_files(tracked)
    commit = run_git(["rev-parse", "HEAD"]).strip() or None
    embed = make_embedder(settings.embeddings_base_url, settings.embeddings_model)
    try:
        summary = index_lib.build(
            settings.index_dir, ROOT, files, commit,
            embed=embed, store=build_index, record_factory=IndexRecord,
        )  # fmt: skip
    except Exception as exc:  # noqa: BLE001 — CLI: понятная ошибка вместо трейсбека
        print(
            f"❌ Сборка индекса остановлена: {type(exc).__name__}: {exc}\n"
            "Прежний индекс (если был) не изменён.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    print(index_lib.format_summary(summary, settings.index_dir))
    return EXIT_OK


def search_context(settings: Settings, queries: list[str]) -> tuple[list, str | None, dict | None]:
    """Фрагменты индекса по запросам. Возвращает (фрагменты, замечание, метаданные индекса)."""
    from rag.index_store import index_exists, search

    if not index_exists(lib.STRATEGY, settings.index_dir):
        return [], lib.NO_INDEX_NOTE, None
    meta = index_lib.read_meta(settings.index_dir)
    if not queries:
        return [], None, meta
    try:
        embed = make_embedder(settings.embeddings_base_url, settings.embeddings_model)
        vectors = embed(queries)
        results = [
            search(lib.STRATEGY, settings.index_dir, vector, settings.top_k) for vector in vectors
        ]
    except Exception as exc:  # noqa: BLE001 — поиск не должен ронять ревью по diff
        print(f"⚠️ Поиск по индексу не удался: {type(exc).__name__}", file=sys.stderr)
        return [], lib.NO_INDEX_NOTE, meta
    merged = lib.merge_results(results, settings.top_k)
    return merged, (None if merged else lib.NO_FRAGMENTS_NOTE), meta


def cmd_review(args: argparse.Namespace) -> int:
    for rev in (args.base, args.head):
        if not lib.is_safe_rev(rev):
            raise ConfigError(f"недопустимая ревизия git: {rev!r}")
    settings = load_settings(need_model=True)
    repo = args.repo or os.getenv("GITHUB_REPOSITORY", "")
    if not args.dry_run and not repo:
        raise ConfigError("не указан репозиторий: --repo или GITHUB_REPOSITORY")

    base_input = dict(
        head_sha=args.head, provider=settings.provider, model=settings.model, repo=repo or None
    )
    extra_unchecked: list[str] = []

    title, body = args.title, args.body
    if title is None and body is None and repo and args.pr:
        try:
            title, body = fetch_pr_meta(repo, args.pr)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Описание PR не получено: {type(exc).__name__}", file=sys.stderr)
            extra_unchecked.append("название и описание PR не получены: ревью только по diff")
    title, body = title or "", body or ""

    diff_text, diff_cut = read_pr_diff(args.base, args.head)
    if diff_cut:
        extra_unchecked.append(
            f"diff прочитан не целиком: первые {MAX_DIFF_READ_CHARS} символов из вывода git"
        )
    files = lib.parse_diff(diff_text)
    contents = {
        f.path: read_file_at(args.head, f.path)
        for f in files
        if not f.is_deleted and lib.noise_reason(f) is None
    }
    selection = lib.select_for_review(files, contents, settings.max_chars)

    if not selection.has_reviewable:
        comment = lib.build_comment(
            lib.CommentInput(
                status=lib.STATUS_NO_CHANGES, noise=list(selection.noise), **base_input
            )
        )
        return _deliver(comment, repo, args, EXIT_OK)

    queries = lib.build_search_queries(selection.included)
    fragments, search_note, meta = search_context(settings, queries)
    unchecked = [*selection.unchecked, *extra_unchecked]
    if search_note:
        unchecked.append(search_note)

    messages = lib.build_messages(selection.included, fragments, title, body, unchecked)
    ask = make_chat(settings)
    index_info = {
        "index_built_at": (meta or {}).get("built_at"),
        "index_commit": (meta or {}).get("commit"),
    }
    try:
        raw, finish = ask(messages)
        parsed = lib.parse_model_response(raw)
        if parsed is None and raw.strip():
            retry = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": lib.RETRY_REMINDER},
            ]
            raw, finish = ask(retry)
            parsed = lib.parse_model_response(raw)
        if parsed is None and not raw.strip():
            # Пустой ответ повторять бессмысленно: у моделей с рассуждением весь лимит уходит
            # на размышления (`finish_reason=length`), повтор даст то же самое.
            raise EmptyAnswerError(finish, settings.max_tokens)
    except Exception as exc:  # noqa: BLE001 — сбой модели → комментарий «ревью не выполнено»
        comment = lib.build_comment(
            lib.CommentInput(
                status=lib.STATUS_FAILED,
                failure_reason=failure_reason(exc, settings),
                **index_info,
                **base_input,
            )
        )
        return _deliver(comment, repo, args, EXIT_FAILED)

    if parsed is None:
        data = lib.CommentInput(
            status=lib.STATUS_OK, raw_text=raw, unchecked=unchecked, **index_info, **base_input
        )
    else:
        data = lib.CommentInput(
            status=lib.STATUS_OK,
            findings=lib.verify_review(parsed, selection.included),
            model_unchecked=parsed.not_checked,
            unchecked=unchecked,
            **index_info,
            **base_input,
        )
    return _deliver(lib.build_comment(data), repo, args, EXIT_OK)


def _deliver(comment: str, repo: str, args: argparse.Namespace, exit_code: int) -> int:
    if args.dry_run:
        print(comment)
        return exit_code
    try:
        action = publish_comment(repo, args.pr, comment)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Комментарий не опубликован: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(f"Комментарий ревью {action} (PR #{args.pr}).")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-ревью pull request")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="собрать индекс ревью из текущей ветки")
    review = sub.add_parser("review", help="выполнить ревью PR и опубликовать комментарий")
    review.add_argument("--pr", type=int, default=0, help="номер PR (для публикации)")
    review.add_argument("--base", default="main", help="база PR: SHA или ветка")
    review.add_argument("--head", required=True, help="SHA коммита PR")
    review.add_argument("--repo", default="", help="owner/repo (по умолчанию GITHUB_REPOSITORY)")
    review.add_argument("--title", default=None, help="название PR вместо запроса к GitHub")
    review.add_argument("--body", default=None, help="описание PR вместо запроса к GitHub")
    review.add_argument("--dry-run", action="store_true", help="напечатать комментарий")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    try:
        if args.command == "index":
            code = cmd_index(args)
        else:
            if not args.dry_run and args.pr <= 0:
                raise ConfigError("для публикации нужен --pr (или используйте --dry-run)")
            code = cmd_review(args)
    except ConfigError as exc:
        print(f"❌ Ошибка настроек: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(f"Готово за {time.monotonic() - started:.1f} с.", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
