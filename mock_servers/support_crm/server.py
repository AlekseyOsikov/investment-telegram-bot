"""Мок-сервер «CRM» для режима /support: MCP по stdio над JSON-файлами с ВЫМЫШЛЕННЫМИ данными.

Ровно три инструмента, все только для чтения: `get_my_profile`, `list_my_tickets`, `get_ticket`.
Каждый принимает `telegram_user_id` (его подставляет бот, не модель) и возвращает данные только
этого пользователя. Демонстрационную запись получает идентификатор из `--demo-user-id`; прочие
идентификаторы — пустой профиль. Файлы данных только читаются.

Запуск: `python mock_servers/support_crm/server.py --demo-user-id <telegram id>`. Не часть бота:
`src/` этот модуль не импортирует, бот запускает его отдельным процессом."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import crm_store
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

DATA_DIR = Path(__file__).resolve().parent / "data"
READ_ONLY = ToolAnnotations(read_only_hint=True)


def load_json(name: str, key: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as handle:
        return list(json.load(handle)[key])


def build_server(demo_user_id: int | None) -> MCPServer:
    users = load_json("users.json", "users")
    tickets = load_json("tickets.json", "tickets")
    server = MCPServer("support-crm-mock")

    @server.tool(annotations=READ_ONLY)
    def get_my_profile(telegram_user_id: int) -> dict[str, Any]:
        """Профиль пользователя поддержки и число его обращений. Неизвестный пользователь —
        пустой профиль."""
        key = crm_store.resolve_user_key(telegram_user_id, demo_user_id)
        return crm_store.get_profile(users, tickets, key)

    @server.tool(annotations=READ_ONLY)
    def list_my_tickets(telegram_user_id: int, status: str | None = None) -> dict[str, Any]:
        """Обращения пользователя (свежие первыми). status — необязательный фильтр:
        open, pending или closed."""
        key = crm_store.resolve_user_key(telegram_user_id, demo_user_id)
        return crm_store.list_tickets(tickets, key, status)

    @server.tool(annotations=READ_ONLY)
    def get_ticket(telegram_user_id: int, ticket_id: str) -> dict[str, Any]:
        """Обращение пользователя по номеру (например, T-1001)."""
        key = crm_store.resolve_user_key(telegram_user_id, demo_user_id)
        return crm_store.get_ticket(tickets, key, ticket_id)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Мок-CRM поддержки (вымышленные данные).")
    parser.add_argument("--demo-user-id", type=int, default=None, help="Telegram id оператора")
    args = parser.parse_args()
    build_server(args.demo_user_id).run("stdio")


if __name__ == "__main__":
    main()
