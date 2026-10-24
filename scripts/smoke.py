"""Smoke-прогон бота без Telegram: `python scripts/smoke.py [сценарий.md ...]`.

Не часть бота. Запускает НАСТОЯЩИЙ `main.main()` (те же обработчики, фильтры и тексты), но
вместо `run_polling` подаёт приложению апдейты из текстовых сценариев `docs/smoke/*.md`, а
исходящие вызовы Bot API не уходят в сеть, а записываются (`Bot._do_post` подменён). Модель,
MCP-серверы и Ollama не запускаются: сценарии выбраны так, чтобы им это не требовалось.
Каталоги данных — во временной папке, рабочие `data/` не трогаются.

На каждый шаг сохраняется «скриншот» диалога (HTML → PNG через headless Chrome; без Chrome
остаётся HTML), в конце — `reports/smoke/<время>/report.md` (или в `--out DIR`). Код выхода 1,
если что-то упало.
Чистые правила (разбор, проверки, отчёт) — `smoke_lib.py`, тесты — `tests/test_smoke_lib.py`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_lib as lib  # noqa: E402

DEFAULT_SCENARIOS = ROOT / "docs" / "smoke"
CHROME_CANDIDATES = ("google-chrome", "chromium", "chromium-browser")


def prepare_environment(workdir: Path) -> None:
    """Окружение ДО импорта `config`: заглушки вместо секретов, выключенные внешние слои,
    данные во временной папке. Значения задаются принудительно — `.env` оператора не должен
    менять ход smoke (dotenv не перезаписывает уже заданные переменные)."""
    forced = {
        "TELEGRAM_BOT_TOKEN": "123456:smoke-token",
        "MAIN_CLIENT": "deepseek",
        "MAIN_MODEL": "smoke-model",
        "DEEPSEEK_API_KEY": "smoke-key",
        "AGENT_HISTORY_DIR": str(workdir / "agent_history"),
        "AGENT_MEMORY_DIR": str(workdir / "smart_agent_memory"),
        "PRICE_WATCH_DB": str(workdir / "watch.sqlite3"),
        "NEWS_DB": str(workdir / "news.sqlite3"),
        "MCP_MOEX_DIR": "",
        "MCP_BYBIT_DIR": "",
        "MCP_GIT_DIR": "",
        "MCP_FILES_ENABLED": "false",
        "MCP_SUPPORT_COMMAND": "",
        "SUPPORT_CRM_MOCK": "false",
        "DEV_ASSISTANT_USER_IDS": "",
        "NEWS_ENABLED": "false",
        "REWRITE_PROVIDER": "",
    }
    os.environ.update(forced)
    # Сети в smoke нет, а прокси из окружения оператора (например, socks://) роняет конструктор
    # HTTP-клиентов провайдеров при импорте.
    for name in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(name, None)
        os.environ.pop(name.lower(), None)


class Captured(Exception):
    """Сигнал из подменённого `run_polling`: приложение собрано, поллинг не нужен."""

    def __init__(self, application) -> None:
        super().__init__("application captured")
        self.application = application


class Transport:
    """Подмена `Bot._do_post`: записывает исходящие сообщения вместо похода в сеть."""

    def __init__(self) -> None:
        self.replies: list[lib.Reply] = []
        self.message_ids: dict[int, int] = {}  # id сообщения бота -> индекс в replies
        self._next_id = 5000

    def reset(self) -> None:
        self.replies = []

    async def post(self, bot, endpoint: str, data: dict, **_kwargs):
        if endpoint == "getMe":
            return {"id": lib.BOT_ID, "is_bot": True, "first_name": "Бот", "username": "smoke_bot"}
        if endpoint in ("sendMessage", "editMessageText"):
            return self._message(endpoint, data)
        return True

    def _message(self, endpoint: str, data: dict) -> dict:
        buttons: list[tuple[str, str]] = []
        markup = data.get("reply_markup")
        for row in getattr(markup, "inline_keyboard", ()) or ():
            buttons += [(button.text, button.callback_data or "") for button in row]
        self.replies.append(lib.Reply(str(data.get("text", "")), buttons))
        if endpoint == "editMessageText":
            message_id = int(data["message_id"])
        else:
            self._next_id += 1
            message_id = self._next_id
        self.message_ids[len(self.replies) - 1] = message_id
        return {
            "message_id": message_id,
            "date": int(datetime.now().timestamp()),
            "chat": {"id": data["chat_id"], "type": "private"},
            "text": str(data.get("text", "")),
        }

    def message_id_of(self, reply: lib.Reply) -> int:
        for index, candidate in enumerate(self.replies):
            if candidate is reply:
                return self.message_ids[index]
        return 0


class ErrorLog(logging.Handler):
    """Копит записи уровня ERROR: необработанное исключение в обработчике — это провал шага,
    даже если бот что-то ответил."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        text = record.getMessage()
        if record.exc_info:
            text += f" ({record.exc_info[0].__name__}: {record.exc_info[1]})"
        self.records.append(text)


def build_application():
    """Настоящая сборка приложения из `main.main()` без поллинга."""
    from telegram.ext import Application

    import main

    def fake_run_polling(self, *args, **kwargs):
        raise Captured(self)

    original = Application.run_polling
    Application.run_polling = fake_run_polling
    try:
        main.main()
    except Captured as captured:
        return captured.application, main
    finally:
        Application.run_polling = original
    raise RuntimeError("main.main() завершился без run_polling")


def find_handler_location(application, command: str) -> str:
    """Где зарегистрирована команда: `модуль.функция (файл:строка)` — подсказка «где искать»."""
    from telegram.ext import CommandHandler, ConversationHandler

    def walk(handler):
        if isinstance(handler, ConversationHandler):
            for entry in handler.entry_points:
                yield from walk(entry)
        elif isinstance(handler, CommandHandler):
            yield handler

    for handlers in application.handlers.values():
        for top in handlers:
            for handler in walk(top):
                if command in handler.commands:
                    callback = handler.callback
                    try:
                        path = Path(inspect.getsourcefile(callback)).relative_to(ROOT)
                        line = inspect.getsourcelines(callback)[1]
                    except (TypeError, OSError, ValueError):
                        return f"{callback.__module__}.{callback.__qualname__}"
                    return f"{callback.__module__}.{callback.__qualname__} ({path}:{line})"
    return f"команда /{command} не зарегистрирована (проверь main.py и флаги в config.py)"


def suspect_for(application, step: lib.Step, last_command: str) -> str:
    if step.action == lib.ACTION_SEND and step.argument.startswith("/"):
        return find_handler_location(application, step.argument.split()[0][1:].split("@")[0])
    if step.action == lib.ACTION_SEND:
        return "main.handle_message (src/main.py) — фильтр обычного текста"
    if last_command:
        return f"кнопка относится к диалогу команды /{last_command}: " + find_handler_location(
            application, last_command
        )
    return "обработчик нажатия кнопки"


def find_chrome() -> str | None:
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def take_screenshot(chrome: str | None, html_path: Path, png_path: Path, height: int) -> str:
    """PNG через headless Chrome; без Chrome или при сбое остаётся HTML-страница."""
    if chrome:
        command = [
            chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            f"--window-size=480,{height}", f"--screenshot={png_path}", html_path.as_uri(),
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if png_path.exists():
            return png_path.name
    return html_path.name


async def run_scenario(application, transport, scenario, out_dir: Path, chrome, counter):
    from telegram import Update

    log = ErrorLog()
    logging.getLogger().addHandler(log)
    transcript: list[tuple[str, lib.Reply]] = []
    results: list[lib.StepResult] = []
    seen_replies: list[lib.Reply] = []  # все ответы сценария: кнопки ищутся среди них
    last_command = ""
    folder = out_dir / _slug(scenario.name)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        for number, step in enumerate(scenario.steps, start=1):
            transport.reset()
            log.records.clear()
            counter[0] += 1
            failures: list[str] = []

            if step.action == lib.ACTION_SEND:
                if step.argument.startswith("/"):
                    last_command = step.argument.split()[0][1:].split("@")[0]
                transcript.append(("user", lib.Reply(step.argument)))
                payload = lib.build_message_update(counter[0], scenario.chat_type, step.argument)
            else:
                found = lib.find_button(seen_replies, step.argument)
                if found is None:
                    payload = None
                    failures.append(f"кнопки «{step.argument}» нет среди ответов бота")
                else:
                    reply, data = found
                    transcript.append(("user", lib.Reply(f"[нажата кнопка «{step.argument}»]")))
                    payload = lib.build_callback_update(
                        counter[0], scenario.chat_type, data,
                        transport.message_id_of(reply), reply.text,
                    )

            if payload is not None:
                update = Update.de_json(payload, application.bot)
                try:
                    await application.process_update(update)
                    for _ in range(5):
                        await asyncio.sleep(0)
                except Exception as exc:  # noqa: BLE001 — любой сбой обработчика = провал шага
                    failures.append(f"необработанное исключение: {type(exc).__name__}: {exc}")

            replies = list(transport.replies)
            seen_replies.extend(replies)
            transcript += [("bot", reply) for reply in replies]
            if payload is not None:
                failures += lib.check_step(step, replies, list(log.records))
            suspect = suspect_for(application, step, last_command) if failures else ""

            page = lib.render_chat_html(scenario.name, step.title, transcript, failures)
            html_path = folder / f"step-{number}.html"
            html_path.write_text(page, encoding="utf-8")
            shot = take_screenshot(
                chrome, html_path, folder / f"step-{number}.png",
                lib.estimate_height(transcript, failures),
            )
            results.append(
                lib.StepResult(
                    step.title, lib.action_text(step), replies, failures,
                    screenshot=f"{folder.name}/{shot}", suspect=suspect,
                )
            )
    finally:
        logging.getLogger().removeHandler(log)
    return lib.ScenarioResult(scenario.name, results)


def _slug(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in name.casefold())
    return "-".join(part for part in cleaned.split("-") if part) or "scenario"


async def run_all(scenarios, out_dir: Path) -> list[lib.ScenarioResult]:
    application, main_module = build_application()
    transport = Transport()
    from telegram import Bot

    async def fake_do_post(bot, endpoint, data, **kwargs):
        return await transport.post(bot, endpoint, data, **kwargs)

    Bot._do_post = fake_do_post
    await application.initialize()
    await main_module._post_init(application)
    chrome = find_chrome()
    counter = [0]
    results = []
    try:
        for scenario in scenarios:
            results.append(
                await run_scenario(application, transport, scenario, out_dir, chrome, counter)
            )
    finally:
        await main_module._post_stop(application)
        await application.shutdown()
    return results


def split_args(argv: list[str]) -> tuple[list[str], Path | None]:
    """Пути сценариев и каталог результатов `--out DIR` (по умолчанию reports/smoke/<время>)."""
    paths: list[str] = []
    out_dir: Path | None = None
    items = iter(argv)
    for item in items:
        if item == "--out":
            value = next(items, "")
            if not value:
                raise SystemExit("--out требует каталог: --out DIR")
            out_dir = Path(value)
        else:
            paths.append(item)
    return paths, out_dir


def collect_paths(args: list[str]) -> list[Path]:
    if args:
        return [Path(arg) for arg in args]
    return sorted(DEFAULT_SCENARIOS.glob("*.md"))


def main() -> int:
    raw_paths, custom_out = split_args(sys.argv[1:])
    paths = [path for path in collect_paths(raw_paths) if path.name != "README.md"]
    if not paths:
        print("Нет сценариев: положи *.md в docs/smoke/ или передай пути аргументами.")
        return 2
    scenarios = []
    for path in paths:
        try:
            scenarios.append(lib.parse_scenario(path.read_text(encoding="utf-8")))
        except lib.ScenarioError as exc:
            print(f"{path}: ошибка в сценарии: {exc}")
            return 2

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = (custom_out or ROOT / "reports" / "smoke" / stamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-") as workdir:
        prepare_environment(Path(workdir))
        logging.disable(logging.NOTSET)
        results = asyncio.run(run_all(scenarios, out_dir))

    report = lib.render_report(results, datetime.now().strftime("%Y-%m-%d %H:%M"))
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    for result in results:
        print(f"{'OK  ' if result.passed else 'FAIL'} {result.name}")
        for step in result.steps:
            for failure in step.failures:
                print(f"     шаг «{step.title}»: {failure}")
    print(f"Отчёт: {out_dir / 'report.md'}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
