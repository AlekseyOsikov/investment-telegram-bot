"""Чистые правила smoke-обвязки (`scripts/smoke.py`): разбор текстовых сценариев, сборка
Telegram-апдейтов, проверка шагов, HTML «скриншота» диалога и итоговый отчёт.

Не знает про Telegram-клиент, `config`, сеть и диск: всё, что можно проверить без бота,
лежит здесь, а `smoke.py` только подаёт апдейты настоящему приложению и пишет файлы.
Тесты — `tests/test_smoke_lib.py`.

Формат сценария (`docs/smoke/*.md`) — обычный markdown; строки вне перечисленных ниже
ключей (пояснения, пустые строки, цитаты) игнорируются:

    # Сценарий: название
    Чат: личный            (или «групповой»; по умолчанию личный)

    ## Шаг: подпись шага
    Отправить: /start      (или «Нажать: <подпись кнопки>»; ровно одно действие на шаг)
    Ожидать: текст         (подстрока ответа бота, без учёта регистра; можно несколько)
    Не ожидать: текст      (подстроки, которых в ответе быть не должно)
    Тишина                 (бот на шаге вообще ничего не отвечает)
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass, field

CHAT_PRIVATE = "private"
CHAT_GROUP = "group"
_CHAT_ALIASES = {"личный": CHAT_PRIVATE, "групповой": CHAT_GROUP}

ACTION_SEND = "send"
ACTION_CLICK = "click"

# Фразы, которых не должно быть ни в одном ответе бота (доменное правило «без гарантий»,
# docs/code-style.md): проверяются на КАЖДОМ шаге каждого сценария, а не только там, где
# сценарий об этом просит.
GUARANTEE = re.compile(
    r"гарантированн|гарантируем|точно вырастет|рекомендуем (?:купить|продать)",
    re.IGNORECASE,
)

BOT_ID = 999_000
USER_ID = 1001
GROUP_ID = -2001


class ScenarioError(ValueError):
    """Сценарий написан неверно (ошибка формы, а не падение проверки)."""


@dataclass(frozen=True)
class Step:
    title: str
    action: str
    argument: str
    expect: tuple[str, ...] = ()
    expect_not: tuple[str, ...] = ()
    silent: bool = False


@dataclass(frozen=True)
class Scenario:
    name: str
    chat_type: str
    steps: tuple[Step, ...]


@dataclass
class Reply:
    """Один исходящий текст бота (отправка или правка сообщения) и его кнопки."""

    text: str
    buttons: list[tuple[str, str]] = field(default_factory=list)  # (подпись, callback_data)


@dataclass
class StepResult:
    title: str
    action_text: str
    replies: list[Reply]
    failures: list[str]
    screenshot: str = ""
    suspect: str = ""

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class ScenarioResult:
    name: str
    steps: list[StepResult]

    @property
    def passed(self) -> bool:
        return all(step.passed for step in self.steps)


# --------------------------------------------------------------------------- #
# Разбор сценария
# --------------------------------------------------------------------------- #

_KEY_RE = re.compile(r"^(Сценарий|Чат|Шаг|Отправить|Нажать|Ожидать|Не ожидать)\s*:\s*(.*)$")


def parse_scenario(text: str) -> Scenario:
    """Текст сценария → `Scenario`; ошибка формы — `ScenarioError` с номером строки."""
    name = ""
    chat_type = CHAT_PRIVATE
    drafts: list[dict] = []

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip().lstrip("#-*> ").strip()  # заголовки, пункты списка и цитаты
        if line == "Тишина":
            _current(drafts, number)["silent"] = True
            continue
        match = _KEY_RE.match(line)
        if match is None:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key == "Сценарий":
            name = value
        elif key == "Чат":
            if value.lower() not in _CHAT_ALIASES:
                raise ScenarioError(
                    f"строка {number}: чат «{value}» — нужно «личный» или «групповой»"
                )
            chat_type = _CHAT_ALIASES[value.lower()]
        elif key == "Шаг":
            drafts.append({"title": value, "actions": [], "expect": [], "expect_not": []})
        elif key in ("Отправить", "Нажать"):
            if not value:
                raise ScenarioError(f"строка {number}: у «{key}» пустое значение")
            kind = ACTION_SEND if key == "Отправить" else ACTION_CLICK
            _current(drafts, number)["actions"].append((kind, value))
        else:
            if not value:
                raise ScenarioError(f"строка {number}: у «{key}» пустое значение")
            target = "expect" if key == "Ожидать" else "expect_not"
            _current(drafts, number)[target].append(value)

    if not name:
        raise ScenarioError("нет строки «# Сценарий: <название>»")
    if not drafts:
        raise ScenarioError("в сценарии нет ни одного шага («## Шаг: <подпись>»)")

    steps = []
    for draft in drafts:
        if len(draft["actions"]) != 1:
            raise ScenarioError(
                f"шаг «{draft['title']}»: нужно ровно одно действие (Отправить или Нажать), "
                f"найдено {len(draft['actions'])}"
            )
        if draft.get("silent") and draft["expect"]:
            raise ScenarioError(f"шаг «{draft['title']}»: «Тишина» несовместима с «Ожидать»")
        kind, argument = draft["actions"][0]
        steps.append(
            Step(
                title=draft["title"],
                action=kind,
                argument=argument,
                expect=tuple(draft["expect"]),
                expect_not=tuple(draft["expect_not"]),
                silent=bool(draft.get("silent")),
            )
        )
    return Scenario(name=name, chat_type=chat_type, steps=tuple(steps))


def _current(drafts: list[dict], number: int) -> dict:
    if not drafts:
        raise ScenarioError(f"строка {number}: действие или проверка до первого «## Шаг:»")
    return drafts[-1]


# --------------------------------------------------------------------------- #
# Апдейты Telegram (JSON в формате Bot API: их разбирает настоящий `Update.de_json`)
# --------------------------------------------------------------------------- #


def chat_id_for(chat_type: str) -> int:
    return GROUP_ID if chat_type == CHAT_GROUP else USER_ID


def _user() -> dict:
    return {"id": USER_ID, "is_bot": False, "first_name": "Смоук"}


def build_message_update(update_id: int, chat_type: str, text: str, now: int | None = None) -> dict:
    """Сообщение пользователя; для команды добавляется сущность bot_command (без неё
    `CommandHandler` сообщение не узнает)."""
    message: dict = {
        "message_id": update_id,
        "date": now if now is not None else int(time.time()),
        "chat": {"id": chat_id_for(chat_type), "type": chat_type},
        "from": _user(),
        "text": text,
    }
    if text.startswith("/"):
        command = text.split(maxsplit=1)[0]
        message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(command)}]
    return {"update_id": update_id, "message": message}


def build_callback_update(
    update_id: int, chat_type: str, data: str, bot_message_id: int, bot_text: str,
    now: int | None = None,
) -> dict:
    """Нажатие инлайн-кнопки под сообщением бота с номером `bot_message_id`."""
    return {
        "update_id": update_id,
        "callback_query": {
            "id": str(update_id),
            "from": _user(),
            "chat_instance": "smoke",
            "data": data,
            "message": {
                "message_id": bot_message_id,
                "date": now if now is not None else int(time.time()),
                "chat": {"id": chat_id_for(chat_type), "type": chat_type},
                "from": {"id": BOT_ID, "is_bot": True, "first_name": "Бот"},
                "text": bot_text,
            },
        },
    }


def find_button(replies: list[Reply], caption: str) -> tuple[Reply, str] | None:
    """Последняя кнопка, в подписи которой есть `caption` (без учёта регистра): сообщение
    и её callback_data. Ищется с конца — нажимают то, что бот показал последним."""
    needle = caption.casefold()
    for reply in reversed(replies):
        for label, data in reply.buttons:
            if needle in label.casefold():
                return reply, data
    return None


# --------------------------------------------------------------------------- #
# Проверка шага
# --------------------------------------------------------------------------- #


def check_step(step: Step, replies: list[Reply], log_errors: list[str]) -> list[str]:
    """Что на шаге пошло не так (пустой список — всё хорошо)."""
    failures: list[str] = []
    # Подписи кнопок — часть того, что видит пользователь, поэтому ищем и в них.
    joined = "\n".join(
        "\n".join([reply.text, *(label for label, _data in reply.buttons)]) for reply in replies
    )
    folded = joined.casefold()

    if step.silent and replies:
        failures.append(f"бот должен был молчать, а ответил: «{_short(joined)}»")
    if not step.silent and not replies:
        failures.append("бот ничего не ответил")
    for needle in step.expect:
        if needle.casefold() not in folded:
            failures.append(f"в ответе нет «{needle}»")
    for needle in step.expect_not:
        if needle.casefold() in folded:
            failures.append(f"в ответе есть запрещённое «{needle}»")
    found = GUARANTEE.search(joined)
    if found:
        failures.append(f"в ответе есть формулировка-гарантия «{found.group(0)}»")
    for error in log_errors:
        failures.append(f"ошибка в логе бота: {_short(error)}")
    return failures


def _short(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def action_text(step: Step) -> str:
    return step.argument if step.action == ACTION_SEND else f"[кнопка] {step.argument}"


# --------------------------------------------------------------------------- #
# HTML-«скриншот» диалога
# --------------------------------------------------------------------------- #

_CSS = """
body{font-family:system-ui,sans-serif;background:#e7ebf0;margin:0;padding:12px;width:456px}
h1{font-size:13px;margin:0 0 8px;color:#333}
.m{max-width:85%;padding:7px 10px;border-radius:10px;margin:6px 0;white-space:pre-wrap;
word-wrap:break-word;font-size:13px;line-height:1.35}
.user{background:#d9fdd3;margin-left:auto}.bot{background:#fff}
.btn{display:inline-block;border:1px solid #6a9fd4;border-radius:6px;padding:2px 8px;
margin:4px 4px 0 0;color:#2a6db0;font-size:12px}
.status{margin-top:10px;padding:6px 10px;border-radius:6px;font-size:12px}
.ok{background:#c8f0c8}.fail{background:#f6c4c4}
"""


def render_chat_html(
    scenario_title: str, step_title: str, transcript: list[tuple[str, Reply]], failures: list[str]
) -> str:
    """Страница-«скриншот»: диалог до текущего шага включительно и статус шага.
    `transcript` — пары («user»|«bot», сообщение)."""
    rows = []
    for who, reply in transcript:
        buttons = "".join(
            f'<span class="btn">{html.escape(label)}</span>' for label, _data in reply.buttons
        )
        rows.append(f'<div class="m {who}">{html.escape(reply.text)}<div>{buttons}</div></div>')
    if failures:
        items = "".join(f"<div>• {html.escape(item)}</div>" for item in failures)
        status = f'<div class="status fail"><b>ПРОВАЛ</b>{items}</div>'
    else:
        status = '<div class="status ok"><b>OK</b></div>'
    return (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        f"<style>{_CSS}</style><body>"
        f"<h1>{html.escape(scenario_title)} — {html.escape(step_title)}</h1>"
        f'{"".join(rows)}{status}</body></html>'
    )


def estimate_height(transcript: list[tuple[str, Reply]], failures: list[str]) -> int:
    """Высота окна для скриншота: по числу строк текста (грубо), в разумных пределах."""
    lines = 4 + len(failures)
    for _who, reply in transcript:
        lines += 2 + sum(max(1, len(part) // 48 + 1) for part in reply.text.split("\n"))
        lines += 1 if reply.buttons else 0
    return max(400, min(6000, lines * 19 + 60))


# --------------------------------------------------------------------------- #
# Итоговый отчёт
# --------------------------------------------------------------------------- #


def render_report(results: list[ScenarioResult], generated: str) -> str:
    """Markdown-отчёт: сводная таблица, затем по сценариям шаги со ссылками на скриншоты и,
    для упавших шагов, причины и подозрительное место."""
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    lines = [
        "# Smoke-отчёт",
        "",
        f"Дата: {generated}. Сценариев пройдено: {passed} из {total}.",
        "",
        "| Сценарий | Результат | Шагов | Упало |",
        "|---|---|---|---|",
    ]
    for result in results:
        failed = sum(1 for step in result.steps if not step.passed)
        mark = "✅ прошёл" if result.passed else "❌ упал"
        lines.append(f"| {result.name} | {mark} | {len(result.steps)} | {failed} |")

    for result in results:
        lines += ["", f"## {result.name}", ""]
        for number, step in enumerate(result.steps, start=1):
            mark = "✅" if step.passed else "❌"
            lines.append(f"{number}. {mark} **{step.title}** — `{step.action_text}`")
            if step.screenshot:
                lines.append(f"   ![шаг {number}]({step.screenshot})")
            for failure in step.failures:
                lines.append(f"   - Что сломалось: {failure}")
            if step.failures and step.suspect:
                lines.append(f"   - Где искать: {step.suspect}")
    lines.append("")
    return "\n".join(lines)
