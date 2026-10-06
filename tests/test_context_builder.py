"""Тесты сборки контекста /smart_agent (agents/context_builder.py): порядок слоёв, выключение
слоёв, окно краткосрочной памяти и обязательные оговорки служебных сообщений. Чистые функции —
без SmartAgent, сети и LLM."""

from agents import context_builder as cb
from agents import memory_state as ms
from agents import rag_context, task_state

SYSTEM = "ОСНОВНОЙ ПРОМПТ"


def make_profile(**overrides) -> dict:
    profile = ms.empty_profile()
    profile.update(overrides)
    return profile


def build(profile, layers=None, pairs=2, tools=None, rag=None):
    return cb.build_context_messages(
        SYSTEM, profile, layers or ms.default_enabled_layers(), pairs, tools, rag
    )


def roles_and_heads(messages) -> list[str]:
    return [f"{m['role']}:{m['content'][:20]}" for m in messages]


def full_profile() -> dict:
    meta = {name: "" for name in ms.PROFILE_FIELDS}
    meta["style"] = "кратко"
    return make_profile(
        meta=meta,
        invariants=[{"text": "без криптовалют", "category": "instruments"}],
        long_term=[{"text": "любит облигации", "source": "user", "created_at": ""}],
        working=task_state.new_task("portfolio", "собрать портфель"),
        short_term=[
            {"role": "user", "content": "в1"},
            {"role": "assistant", "content": "о1"},
        ],
    )


def test_no_active_profile_gives_only_system_prompt():
    assert build(None) == [{"role": "system", "content": SYSTEM}]


def test_empty_profile_gives_only_system_prompt():
    assert build(make_profile()) == [{"role": "system", "content": SYSTEM}]


def test_layer_order_is_invariants_profile_tools_rag_long_term_task_dialog():
    tools = [{"role": "system", "content": "TOOLS"}]
    rag = [{"role": "system", "content": "RAG"}]
    messages = build(full_profile(), tools=tools, rag=rag)
    contents = [m["content"] for m in messages]
    assert contents[0] == SYSTEM
    markers = [
        "без криптовалют",  # инварианты
        "Профиль пользователя",
        "TOOLS",
        "RAG",
        "Долговременная память",
        None,  # сообщение автомата задачи
        "в1",
        "о1",
    ]
    positions = []
    for marker in markers:
        if marker is None:
            continue
        positions.append(next(i for i, c in enumerate(contents) if marker in c))
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    # задача — между долговременной памятью и диалогом
    memory_index = next(i for i, c in enumerate(contents) if "Долговременная память" in c)
    dialog_index = next(i for i, c in enumerate(contents) if c == "в1")
    assert dialog_index - memory_index == 2
    assert len(messages) == 9  # система, 6 служебных, 2 реплики


def test_disabled_layers_are_left_out_but_data_stays():
    profile = full_profile()
    layers = {name: False for name in ms.ALL_LAYERS}
    assert build(profile, layers) == [{"role": "system", "content": SYSTEM}]
    assert profile["long_term"] and profile["invariants"]  # данные не тронуты


def test_each_layer_toggles_independently():
    profile = full_profile()
    base = ms.default_enabled_layers()
    for layer, marker in (
        (ms.LAYER_INVARIANTS, "без криптовалют"),
        (ms.LAYER_PROFILE, "Профиль пользователя"),
        (ms.LAYER_LONG_TERM, "Долговременная память"),
        (ms.LAYER_SHORT_TERM, "в1"),
    ):
        on = [m["content"] for m in build(profile, base)]
        off = [m["content"] for m in build(profile, {**base, layer: False})]
        assert any(marker in c for c in on)
        assert not any(marker in c for c in off)


def test_short_term_window_keeps_last_pairs_and_zero_means_none():
    short_term = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(6)
    ]
    profile = make_profile(short_term=short_term)
    assert [m["content"] for m in build(profile, pairs=1)[1:]] == ["4", "5"]
    assert [m["content"] for m in build(profile, pairs=0)[1:]] == []


def test_profile_meta_message_only_filled_fields_and_keeps_warning_clause():
    assert cb.profile_meta_message({name: "" for name in ms.PROFILE_FIELDS}) is None
    message = cb.profile_meta_message({"style": "кратко", "format": ""})
    assert message["role"] == "system"
    assert "- Стиль общения: кратко" in message["content"]
    assert "Формат ответа" not in message["content"]
    # оговорка: профиль не ослабляет обязательные предупреждения
    assert "не отменяют" in message["content"]
    assert "предупреждения" in message["content"]


def test_rag_messages_depend_on_materials():
    assert cb.rag_messages(rag_context.Materials()) is None
    searched_empty = cb.rag_messages(rag_context.Materials(searched=True))
    assert searched_empty == [rag_context.build_no_materials_message()]
    with_chunks = cb.rag_messages(rag_context.Materials(chunks=[object()], searched=True))
    assert with_chunks == [rag_context.build_rules_message()]
