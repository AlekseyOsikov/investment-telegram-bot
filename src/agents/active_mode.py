"""Общий трекер активного режима `/agent`/`/agent_compare` на чат.

`/agent` (agent_command.py) и `/agent_compare` (compare_command.py) — два независимых
`ConversationHandler`, и без явной проверки ничто не мешает пользователю оказаться
"внутри" обоих одновременно (у каждого ConversationHandler своё отслеживание состояния
диалога, независимое от другого). Это осознанно нежелательно: `/agent` работает с
одной стратегией на чат, а `/agent_compare` — с тремя параллельными "теневыми" агентами
поверх того же chat_id, и смешение вопросов между двумя режимами не имеет понятного
пользователю смысла. Поэтому оба модуля перед входом проверяют, не занят ли чат другим
режимом, и выставляют/снимают отметку здесь — в отдельном модуле, а не в
agent_command.py или compare_command.py, чтобы не создавать цикл импорта между ними
(compare_command.py и так импортирует часть общих хелперов из agent_command.py).

Хранится только в памяти процесса (не персистится на диск) — если бот перезапустится
посреди диалога в одном из режимов, отметка потеряется, и следующий /agent или
/agent_compare отработает как обычный вход; ничего критичного не ломается, т.к. сама
история (Agent) переживает перезапуск независимо от этой отметки.
"""

from __future__ import annotations

AGENT_MODE = "agent"
COMPARE_MODE = "compare"

_active_mode: dict[int, str] = {}


def get_active_mode(chat_id: int) -> str | None:
    return _active_mode.get(chat_id)


def set_active_mode(chat_id: int, mode: str) -> None:
    _active_mode[chat_id] = mode


def clear_active_mode(chat_id: int) -> None:
    _active_mode.pop(chat_id, None)
