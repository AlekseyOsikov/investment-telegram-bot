"""Тесты мок-CRM поддержки (mock_servers/support_crm/crm_store.py и файлы данных): схема, изоляция,
пустой профиль, отсутствие чувствительных сведений. Без процессов и сети."""

import json
import re
from pathlib import Path

import crm_store

DATA = Path(__file__).resolve().parent.parent / "mock_servers" / "support_crm" / "data"
DEMO_ID = 111


def load(name, key):
    return json.loads((DATA / name).read_text(encoding="utf-8"))[key]


USERS = load("users.json", "users")
TICKETS = load("tickets.json", "tickets")

TICKET_FIELDS = {
    "ticket_id", "user_key", "subject", "bot_feature", "status",
    "created", "updated", "error_text", "events",
}
USER_FIELDS = {"user_key", "display_name", "plan", "registered"}


def demo_key():
    return crm_store.resolve_user_key(DEMO_ID, DEMO_ID)


# --- данные ----------------------------------------------------------------------------------


def test_data_schema():
    assert all(set(u) == USER_FIELDS for u in USERS)
    for ticket in TICKETS:
        assert set(ticket) == TICKET_FIELDS
        assert ticket["status"] in crm_store.STATUSES
        assert all(set(e) == {"date", "text"} for e in ticket["events"])
    keys = {u["user_key"] for u in USERS}
    assert {t["user_key"] for t in TICKETS} <= keys
    assert len({t["ticket_id"] for t in TICKETS}) == len(TICKETS)


def test_demo_has_three_or_four_tickets_and_others_have_own():
    owners = [t["user_key"] for t in TICKETS]
    assert 3 <= owners.count("demo") <= 4
    assert any(owner != "demo" for owner in owners)


def test_no_sensitive_data_in_files():
    raw = (DATA / "users.json").read_text(encoding="utf-8") + (DATA / "tickets.json").read_text(
        encoding="utf-8"
    )
    assert "@" not in raw  # e-mail
    assert not re.search(r"\d{6,}", raw)  # номера счетов, карт, телефонов
    assert not re.search(r"\d[\d\s]*(₽|руб|\$|USD|RUB)", raw)  # суммы


def test_there_is_a_ticket_with_instruction_text_for_demo():
    demo_texts = [
        e["text"].lower()
        for t in TICKETS
        if t["user_key"] == "demo"
        for e in t["events"]
    ] + [t["error_text"].lower() for t in TICKETS if t["user_key"] == "demo"]
    assert any("игнорируй" in text and "инструкции" in text for text in demo_texts)


# --- сопоставление ---------------------------------------------------------------------------


def test_demo_id_maps_to_demo_user():
    assert crm_store.resolve_user_key(DEMO_ID, DEMO_ID) == "demo"


def test_other_id_is_unknown():
    assert crm_store.resolve_user_key(222, DEMO_ID) is None


def test_no_demo_id_means_nobody_gets_demo_data():
    assert crm_store.resolve_user_key(DEMO_ID, None) is None
    assert crm_store.resolve_user_key(0, None) is None


def test_non_integer_user_id_is_unknown():
    assert crm_store.resolve_user_key("111", DEMO_ID) is None
    assert crm_store.resolve_user_key(True, 1) is None
    assert crm_store.resolve_user_key(None, DEMO_ID) is None


# --- профиль и список ------------------------------------------------------------------------


def test_profile_of_demo():
    profile = crm_store.get_profile(USERS, TICKETS, demo_key())
    assert profile["known"] is True
    assert profile["tickets_total"] == sum(1 for t in TICKETS if t["user_key"] == "demo")
    assert profile["tickets_open"] == sum(
        1 for t in TICKETS if t["user_key"] == "demo" and t["status"] != "closed"
    )


def test_empty_profile_for_unknown_user():
    assert crm_store.get_profile(USERS, TICKETS, None) == {
        "known": False,
        "tickets_total": 0,
        "tickets_open": 0,
    }


def test_list_returns_only_own_tickets_without_user_key():
    result = crm_store.list_tickets(TICKETS, demo_key())
    ids = {t["ticket_id"] for t in result["tickets"]}
    assert ids == {t["ticket_id"] for t in TICKETS if t["user_key"] == "demo"}
    assert all("user_key" not in t for t in result["tickets"])


def test_list_is_sorted_freshest_first():
    updated = [t["updated"] for t in crm_store.list_tickets(TICKETS, demo_key())["tickets"]]
    assert updated == sorted(updated, reverse=True)


def test_list_filters_by_status():
    result = crm_store.list_tickets(TICKETS, demo_key(), "closed")
    assert result["tickets"] and all(t["status"] == "closed" for t in result["tickets"])


def test_list_unknown_status_is_message_not_exception():
    result = crm_store.list_tickets(TICKETS, demo_key(), "bogus")
    assert result["tickets"] == [] and result["error"]


def test_list_for_unknown_user_is_empty():
    assert crm_store.list_tickets(TICKETS, None) == {"tickets": []}


# --- обращение по номеру: изоляция -----------------------------------------------------------


def test_own_ticket_found_case_insensitive():
    result = crm_store.get_ticket(TICKETS, demo_key(), " t-1001 ")
    assert result["found"] is True and result["ticket"]["ticket_id"] == "T-1001"
    assert "user_key" not in result["ticket"]


def test_foreign_ticket_is_indistinguishable_from_missing():
    foreign_id = next(t["ticket_id"] for t in TICKETS if t["user_key"] != "demo")
    foreign = crm_store.get_ticket(TICKETS, demo_key(), foreign_id)
    missing = crm_store.get_ticket(TICKETS, demo_key(), "T-9999")
    assert foreign == missing
    assert foreign["found"] is False


def test_unknown_user_gets_not_found_even_for_real_number():
    assert crm_store.get_ticket(TICKETS, None, "T-1001") == crm_store.get_ticket(
        TICKETS, None, "T-9999"
    )
    assert crm_store.get_ticket(TICKETS, None, "T-1001")["found"] is False


def test_functions_do_not_mutate_data():
    before = json.dumps(TICKETS, sort_keys=True)
    crm_store.list_tickets(TICKETS, demo_key())
    crm_store.get_ticket(TICKETS, demo_key(), "T-1001")
    crm_store.get_profile(USERS, TICKETS, demo_key())
    assert json.dumps(TICKETS, sort_keys=True) == before
