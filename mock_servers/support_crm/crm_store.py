"""Чистая выборка данных мок-CRM (спека support-crm-mock): профиль, список обращений, обращение по
номеру. Без MCP, сети и записи на диск — данные передаются аргументами, поэтому изоляция
проверяется тестами (tests/test_support_crm_mock.py).

Сопоставление: Telegram-идентификатор оператора (`demo_user_id`) получает записи пользователя
`DEMO_USER_KEY`, все прочие идентификаторы — пустой профиль. Не задан `demo_user_id` — демо-данные
не получает никто. Чужое и несуществующее обращение неотличимы."""

from __future__ import annotations

from typing import Any

DEMO_USER_KEY = "demo"
STATUSES = ("open", "pending", "closed")

NOT_FOUND_MESSAGE = "Обращение с таким номером не найдено."
UNKNOWN_STATUS_MESSAGE = "Неизвестный статус. Допустимые значения: " + ", ".join(STATUSES) + "."


def resolve_user_key(telegram_user_id: Any, demo_user_id: int | None) -> str | None:
    """Ключ записи пользователя в данных или None (пользователь CRM не известен)."""
    if demo_user_id is None:
        return None
    if isinstance(telegram_user_id, bool) or not isinstance(telegram_user_id, int):
        return None
    return DEMO_USER_KEY if telegram_user_id == demo_user_id else None


def get_profile(users: list[dict], tickets: list[dict], user_key: str | None) -> dict:
    """Профиль пользователя. Неизвестный — пустой профиль (`known=False`), а не ошибка."""
    record = next((u for u in users if u.get("user_key") == user_key), None) if user_key else None
    if record is None:
        return {"known": False, "tickets_total": 0, "tickets_open": 0}
    own = [t for t in tickets if t.get("user_key") == user_key]
    return {
        "known": True,
        "display_name": record.get("display_name", ""),
        "plan": record.get("plan", ""),
        "registered": record.get("registered", ""),
        "tickets_total": len(own),
        "tickets_open": sum(1 for t in own if t.get("status") != "closed"),
    }


def _public(ticket: dict) -> dict:
    """Обращение для ответа: без служебного `user_key`."""
    return {key: value for key, value in ticket.items() if key != "user_key"}


def list_tickets(tickets: list[dict], user_key: str | None, status: str | None = None) -> dict:
    """Обращения ТОЛЬКО этого пользователя, самые свежие первыми. `status` — необязательный
    фильтр; неизвестное значение — сообщение, а не исключение."""
    if status is not None and status not in STATUSES:
        return {"tickets": [], "error": UNKNOWN_STATUS_MESSAGE}
    if user_key is None:
        return {"tickets": []}
    own = [
        t
        for t in tickets
        if t.get("user_key") == user_key and (status is None or t.get("status") == status)
    ]
    own.sort(key=lambda t: (t.get("updated", ""), t.get("ticket_id", "")), reverse=True)
    return {"tickets": [_public(t) for t in own]}


def get_ticket(tickets: list[dict], user_key: str | None, ticket_id: str) -> dict:
    """Обращение по номеру. Чужое, несуществующее и запрос без известного пользователя дают
    один и тот же ответ — наличие чужих обращений не раскрывается."""
    wanted = (ticket_id or "").strip().upper()
    if user_key is not None:
        for ticket in tickets:
            same_id = str(ticket.get("ticket_id", "")).upper() == wanted
            if ticket.get("user_key") == user_key and same_id:
                return {"found": True, "ticket": _public(ticket)}
    return {"found": False, "message": NOT_FOUND_MESSAGE}
