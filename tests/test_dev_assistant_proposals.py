"""Тесты чистых правил предложений правок режима /dev (dev_assistant/proposals.py): реестр,
владелец, callback_data, сводка, вывод diff, имя патча. Без Telegram и сети."""

from datetime import datetime, timezone

import pytest

from dev_assistant import proposals

PID = "0123456789ab"


def server_data(**overrides):
    data = {
        "proposal_id": PID, "path": "README.md", "creates": False, "added": 2, "removed": 1,
        "summary": "обновил раздел", "diff": "diff --git a/README.md b/README.md\n+x\n-y\n",
    }
    data.update(overrides)
    return data


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def registry(ttl=60, clock=None, max_items=100):
    return proposals.ProposalRegistry(ttl, clock or Clock(), max_items)


def test_parse_server_proposal_accepts_valid_and_rejects_broken():
    assert proposals.parse_server_proposal(server_data()) is not None
    for bad in (
        None, [], server_data(proposal_id="../x"), server_data(proposal_id=5),
        server_data(path=1), server_data(diff=""), server_data(diff=None),
        server_data(added="2"), server_data(removed=True),
    ):
        assert proposals.parse_server_proposal(bad) is None


def test_register_get_pop_and_double_pop():
    reg = registry()
    record = reg.register(10, 20, server_data())
    assert record.chat_id == 10 and record.user_id == 20 and record.path == "README.md"
    assert reg.get(PID) == record and len(reg) == 1
    assert reg.pop(PID) == record
    assert reg.pop(PID) is None and reg.get(PID) is None


def test_registry_ttl():
    clock = Clock()
    reg = registry(ttl=60, clock=clock)
    reg.register(1, 1, server_data())
    clock.now += 61
    assert reg.get(PID) is None and len(reg) == 0


def test_registry_eviction_and_clear_chat():
    reg = registry(max_items=2)
    for index, chat in enumerate((1, 2, 1)):
        reg.register(chat, chat, server_data(proposal_id=f"{index:012x}"))
    assert len(reg) == 2 and reg.get(f"{0:012x}") is None
    assert reg.clear_chat(1) == 1 and len(reg) == 1


def test_is_owner_requires_both_chat_and_user():
    record = registry().register(10, 20, server_data())
    assert proposals.is_owner(record, 10, 20)
    assert not proposals.is_owner(record, 10, 21)
    assert not proposals.is_owner(record, 11, 20)
    assert not proposals.is_owner(record, None, None)


def test_callback_data_roundtrip_and_size():
    for action in (proposals.ACTION_APPLY, proposals.ACTION_DISCARD):
        data = proposals.build_callback_data(action, PID)
        assert len(data.encode()) <= 64
        assert proposals.parse_callback_data(data) == (action, PID)


@pytest.mark.parametrize(
    "data",
    [None, 5, "", "dev", "dev:apply", "dev:apply:", "dev:apply:XYZ", "dev:drop:" + PID,
     "x:apply:" + PID, "dev:apply:" + PID + ":x", "dev:apply:../etc/passwd"],
)
def test_parse_callback_data_rejects_garbage(data):
    assert proposals.parse_callback_data(data) is None


def test_build_callback_data_validates():
    with pytest.raises(ValueError):
        proposals.build_callback_data("drop", PID)
    with pytest.raises(ValueError):
        proposals.build_callback_data("apply", "short")


def make_proposal(**overrides):
    return registry().register(1, 1, server_data(**overrides))


def test_summary_mentions_file_counts_and_kind():
    text = proposals.diff_summary(make_proposal())
    assert "README.md" in text and "+2" in text and "−1" in text and "обновил раздел" in text
    assert "новый файл" in proposals.diff_summary(make_proposal(creates=True))


def test_render_short_diff_inline_escaped():
    proposal = make_proposal(diff="+<b>&\n")
    rendered = proposals.render_proposal(proposal, 4000)
    assert rendered.document_name is None
    assert "<pre>+&lt;b&gt;&amp;\n</pre>" in rendered.text


def test_render_summary_is_escaped():
    rendered = proposals.render_proposal(make_proposal(summary="<script>"), 4000)
    assert "<script>" not in rendered.text and "&lt;script&gt;" in rendered.text


def test_render_long_diff_goes_to_document_with_summary():
    proposal = make_proposal(diff="+" + "x" * 5000 + "\n")
    rendered = proposals.render_proposal(proposal, 4000)
    assert rendered.document_name == f"{PID}.patch"
    assert "<pre>" not in rendered.text and "README.md" in rendered.text
    assert len(rendered.text) <= 4000


def test_render_boundary_counts_escaped_length():
    # символ & после экранирования занимает 5 знаков: сырой diff помещается, экранированный — нет
    proposal = make_proposal(diff="&" * 900)
    assert proposals.render_proposal(proposal, 4000).document_name is not None
    assert proposals.render_proposal(proposal, 8000).document_name is None


def test_patch_filename_utc():
    moment = datetime(2026, 10, 17, 12, 30, 5, tzinfo=timezone.utc)
    assert proposals.patch_filename(PID, moment) == f"20261017T123005Z-{PID}.patch"


def test_applied_text_has_reindex_reminder_and_patch():
    text = proposals.applied_text("README.md", 2, 1, "/srv/data/dev_patches/p.patch")
    assert "README.md" in text and "make dev-index" in text
    assert "git apply -R /srv/data/dev_patches/p.patch" in text
    assert proposals.PATCH_NOT_SAVED_TEXT in proposals.applied_text("README.md", 2, 1, None)


def test_tool_error_message_strips_sdk_prefix():
    assert proposals.tool_error_message(
        "Error executing tool apply_change: Исходное содержимое изменилось.", "apply_change"
    ) == "Исходное содержимое изменилось."
    assert proposals.tool_error_message("Error executing tool apply_change", "apply_change")


# --- Дополнение: слишком длинный callback_data и текст отказа ----------------------


def test_callback_data_longer_than_telegram_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(proposals, "CALLBACK_MAX_BYTES", 10)
    with pytest.raises(ValueError, match="64 байт"):
        proposals.build_callback_data(proposals.ACTION_APPLY, "0123456789ab")


def test_callback_data_has_prefix_action_and_id_within_limit():
    data = proposals.build_callback_data(proposals.ACTION_DISCARD, "0123456789ab")
    assert data == f"dev:{proposals.ACTION_DISCARD}:0123456789ab"
    assert len(data.encode("utf-8")) <= 64


@pytest.mark.parametrize(
    "proposal_id", ["", "XYZ", "0123456789a", "0123456789abc", "0123456789AB"]
)
def test_callback_data_rejects_bad_ids(proposal_id):
    with pytest.raises(ValueError):
        proposals.build_callback_data(proposals.ACTION_APPLY, proposal_id)


def test_callback_data_rejects_unknown_action():
    with pytest.raises(ValueError):
        proposals.build_callback_data("delete", "0123456789ab")


def test_failure_text_names_reason_and_says_file_unchanged():
    text = proposals.failure_text("файл изменился")
    assert text == "⚠️ Не применено: файл изменился\nФайл не менялся."
