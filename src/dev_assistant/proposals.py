"""Чистые правила предложений правок режима `/dev` (design.md изменения add-dev-project-files,
решения 3, 7, 8): реестр «предложение ↔ чат ↔ пользователь» в памяти процесса, `callback_data`
кнопок, сводка и вывод diff, имя патч-файла. Не знает ни про Telegram, ни про `config`, ни про
MCP — тесты tests/test_dev_assistant_proposals.py.

Реестр живёт только в памяти: после перезапуска бота кнопки отвечают «предложение устарело», а
файл предложения на диске дожидается очистки по сроку жизни на стороне сервера."""

from __future__ import annotations

import html
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

ACTION_APPLY = "apply"
ACTION_DISCARD = "discard"
CALLBACK_PREFIX = "dev"
CALLBACK_PATTERN = rf"^{CALLBACK_PREFIX}:({ACTION_APPLY}|{ACTION_DISCARD}):"
CALLBACK_MAX_BYTES = 64
_ID_RE = re.compile(r"^[0-9a-f]{12}$")

APPLY_LABEL = "✅ Применить"
DISCARD_LABEL = "✖️ Отклонить"

STALE_TEXT = "⚠️ Предложение устарело (бот перезапускался или срок вышел). Повтори задачу."
DISCARDED_TEXT = "Предложение отклонено, файл не менялся."
INDEX_REMINDER = (
    "ℹ️ Индекс документации режима не обновляется сам: чтобы ассистент увидел правку, "
    "пересобери его командой `make dev-index`."
)
SERVER_FAILURE_TEXT = "⚠️ Не удалось связаться с файловым сервером: файл не менялся."
UNKNOWN_STATE_TEXT = (
    "⚠️ Связь с файловым сервером оборвалась, результат неизвестен: проверь файл (`git diff`) "
    "и при необходимости повтори задачу."
)
PATCH_NOT_SAVED_TEXT = "Патч сохранить не удалось (см. журнал бота)."
MAX_REGISTRY_ITEMS = 100


@dataclass(frozen=True)
class Proposal:
    """Предложение правки, подготовленное моделью и показанное пользователю."""

    id: str
    chat_id: int
    user_id: int
    path: str
    creates: bool
    added: int
    removed: int
    summary: str
    diff: str
    created_at: float


def is_valid_id(proposal_id: object) -> bool:
    return isinstance(proposal_id, str) and _ID_RE.match(proposal_id) is not None


def parse_server_proposal(data: object) -> dict | None:
    """Проверяет ответ `propose_change` сервера; `None` — ответ не годится для показа."""
    if not isinstance(data, dict):
        return None
    if not is_valid_id(data.get("proposal_id")):
        return None
    if not isinstance(data.get("path"), str) or not isinstance(data.get("diff"), str):
        return None
    if not data["diff"]:
        return None
    for key in ("added", "removed"):
        if not isinstance(data.get(key), int) or isinstance(data.get(key), bool):
            return None
    return data


class ProposalRegistry:
    """Реестр ожидающих предложений: id → `Proposal`. Срок жизни совпадает с серверным; при
    переполнении вытесняются самые старые. Потокобезопасность: операции атомарны для GIL
    (словарь), дополнительных замков не требуется."""

    def __init__(
        self,
        ttl_seconds: float,
        clock: Callable[[], float] = time.time,
        max_items: int = MAX_REGISTRY_ITEMS,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._max = max_items
        self._items: OrderedDict[str, Proposal] = OrderedDict()

    def register(self, chat_id: int, user_id: int, data: dict) -> Proposal:
        """Регистрирует предложение из проверенного ответа сервера (`parse_server_proposal`)."""
        proposal = Proposal(
            id=data["proposal_id"],
            chat_id=chat_id,
            user_id=user_id,
            path=data["path"],
            creates=bool(data.get("creates")),
            added=data["added"],
            removed=data["removed"],
            summary=str(data.get("summary") or ""),
            diff=data["diff"],
            created_at=self._clock(),
        )
        self._items[proposal.id] = proposal
        while len(self._items) > self._max:
            self._items.popitem(last=False)
        return proposal

    def get(self, proposal_id: str) -> Proposal | None:
        proposal = self._items.get(proposal_id)
        if proposal is None:
            return None
        if self._clock() - proposal.created_at > self._ttl:
            self._items.pop(proposal_id, None)
            return None
        return proposal

    def pop(self, proposal_id: str) -> Proposal | None:
        """Вынимает предложение (защита от двойного нажатия: второй вызов получит `None`)."""
        proposal = self.get(proposal_id)
        if proposal is not None:
            self._items.pop(proposal_id, None)
        return proposal

    def clear_chat(self, chat_id: int) -> int:
        """Забывает предложения чата (выход из режима); возвращает число забытых."""
        ids = [pid for pid, p in self._items.items() if p.chat_id == chat_id]
        for pid in ids:
            self._items.pop(pid, None)
        return len(ids)

    def __len__(self) -> int:
        return len(self._items)


def is_owner(proposal: Proposal, chat_id: int | None, user_id: int | None) -> bool:
    """Нажимает тот, кто задал вопрос, в том же чате (по id пользователя, не только чата)."""
    return proposal.chat_id == chat_id and proposal.user_id == user_id


def build_callback_data(action: str, proposal_id: str) -> str:
    if action not in (ACTION_APPLY, ACTION_DISCARD) or not is_valid_id(proposal_id):
        raise ValueError("недопустимые действие или идентификатор предложения")
    data = f"{CALLBACK_PREFIX}:{action}:{proposal_id}"
    if len(data.encode("utf-8")) > CALLBACK_MAX_BYTES:
        raise ValueError("callback_data длиннее 64 байт")
    return data


def parse_callback_data(data: object) -> tuple[str, str] | None:
    """`(действие, id)` либо `None` для постороннего или повреждённого `callback_data`."""
    if not isinstance(data, str):
        return None
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
        return None
    action, proposal_id = parts[1], parts[2]
    if action not in (ACTION_APPLY, ACTION_DISCARD) or not is_valid_id(proposal_id):
        return None
    return action, proposal_id


def diff_summary(proposal: Proposal) -> str:
    """Заголовок предложения: файл, создание/правка, число добавленных и удалённых строк."""
    kind = "новый файл" if proposal.creates else "правка"
    text = f"✏️ {proposal.path} — {kind}: +{proposal.added} −{proposal.removed}"
    if proposal.summary:
        text += f"\n{proposal.summary}"
    return text


@dataclass(frozen=True)
class RenderedProposal:
    """Что отправить пользователю: HTML-текст и, если diff не помещается, имя файла-патча."""

    text: str
    document_name: str | None = None


def render_proposal(proposal: Proposal, limit: int) -> RenderedProposal:
    """Сообщение с diff в блоке `<pre>` в пределах `limit` символов (после экранирования HTML);
    длиннее — только сводка в тексте, а diff уходит документом `.patch`."""
    header = html.escape(diff_summary(proposal))
    inline = f"{header}\n\n<pre>{html.escape(proposal.diff)}</pre>"
    if len(inline) <= limit:
        return RenderedProposal(inline)
    note = "\n\nDiff не помещается в сообщение — отправляю файлом."
    return RenderedProposal(header + note, document_name=f"{proposal.id}.patch")


def patch_filename(proposal_id: str, now: datetime | None = None) -> str:
    """Имя сохранённого патча: `<UTC-время>-<id>.patch`."""
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{proposal_id}.patch"


def applied_text(path: str, added: int, removed: int, patch_path: str | None) -> str:
    """`patch_path` — полный путь к сохранённому патчу: команда отката дана целиком."""
    lines = [f"✅ Применено: {path} (+{added} −{removed})."]
    if patch_path:
        lines.append(f"Патч сохранён: {patch_path}")
        lines.append(f"Откат (из корня репозитория): git apply -R {patch_path}")
    else:
        lines.append(PATCH_NOT_SAVED_TEXT)
    lines.append(INDEX_REMINDER)
    return "\n".join(lines)


def tool_error_message(text: str, tool: str) -> str:
    """Текст отказа сервера без служебного префикса SDK («Error executing tool X: …»)."""
    prefix = f"Error executing tool {tool}"
    message = text.removeprefix(prefix).removeprefix(":").strip()
    return message or "внутренняя ошибка сервера"


def failure_text(message: str) -> str:
    return f"⚠️ Не применено: {message}\nФайл не менялся."
