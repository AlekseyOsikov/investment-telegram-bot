"""Общий трекер активного режима `/agent`/`/agent_compare`/`/smart_agent` на чат.

`/agent` (agent_command.py), `/agent_compare` (compare_command.py) и `/smart_agent`
(smart_agent_command.py) — три независимых `ConversationHandler`, и без явной проверки
ничто не мешает пользователю оказаться "внутри" нескольких одновременно (у каждого
ConversationHandler своё отслеживание состояния диалога, независимое от других). Это
осознанно нежелательно: у каждого режима свой смысл (одна стратегия на чат / три
параллельных "теневых" агента / три явно разделённых слоя памяти), и смешение вопросов
между режимами не имеет понятного пользователю смысла. Поэтому все три модуля перед
входом проверяют, не занят ли чат другим режимом, и выставляют/снимают отметку здесь —
в отдельном модуле, а не в одном из трёх, чтобы не создавать цикл импорта между ними
(compare_command.py и так импортирует часть общих хелперов из agent_command.py).

Хранится только в памяти процесса (не персистится на диск) — если бот перезапустится
посреди диалога в одном из режимов, отметка потеряется, и следующий вход в любой из
команд отработает как обычный; ничего критичного не ломается, т.к. сама история
(Agent/SmartAgent) переживает перезапуск независимо от этой отметки.
"""

from __future__ import annotations

AGENT_MODE = "agent"
COMPARE_MODE = "compare"
SMART_AGENT_MODE = "smart_agent"

_active_mode: dict[int, str] = {}


def get_active_mode(chat_id: int) -> str | None:
    return _active_mode.get(chat_id)


def set_active_mode(chat_id: int, mode: str) -> None:
    _active_mode[chat_id] = mode


def clear_active_mode(chat_id: int) -> None:
    _active_mode.pop(chat_id, None)
