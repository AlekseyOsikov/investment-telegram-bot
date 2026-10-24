"""Тесты стратегий контекста `/agent` (agents/context_strategies.py): сборка сообщений между
системным промптом и вопросом, побочные действия до и после хода, реестр STRATEGIES.

Стратегии читают состояние агента, поэтому вместо настоящего `Agent` (клиент, диск, модель)
подставляется простой держатель данных с теми же защищёнными полями; вызовы «обновить факты» и
«обновить сводку» он только записывает. Без сети и модели.
"""

import pytest

from agents import context_strategies as cs


class FakeAgent:
    """Данные ветки и счётчики; реальная логика окна — как у Agent._recent_window."""

    def __init__(self, messages=(), facts=None, summary="", summarized_pairs=0, recent_pairs=2):
        self._messages = list(messages)
        self._facts = dict(facts or {})
        self._summary = summary
        self._summarized_pairs = summarized_pairs
        self._recent_pairs = recent_pairs
        self.calls: list[str] = []

    def _active_messages(self):
        return self._messages

    def _recent_window(self, messages):
        return messages[-2 * self._recent_pairs :] if self._recent_pairs else []

    def _update_facts(self):
        self.calls.append("facts")

    def _maybe_update_summary(self):
        self.calls.append("summary")


def _dialog(pairs: int) -> list[dict]:
    messages = []
    for i in range(1, pairs + 1):
        messages.append({"role": "user", "content": f"вопрос {i}"})
        messages.append({"role": "assistant", "content": f"ответ {i}"})
    return messages


def _contents(messages) -> list[str]:
    return [m["content"] for m in messages]


# --- реестр -------------------------------------------------------------------------------


def test_registry_has_four_strategies_keyed_by_their_names():
    assert set(cs.STRATEGIES) == {"sliding_window", "sticky_facts", "branching", "summary"}
    for name, strategy in cs.STRATEGIES.items():
        assert strategy.name == name


def test_registry_matches_configured_strategy_labels():
    from config import AGENT_STRATEGY_LABELS

    assert set(cs.STRATEGIES) == set(AGENT_STRATEGY_LABELS)


def test_base_class_is_abstract_but_hooks_default_to_noop():
    with pytest.raises(TypeError):
        cs.ContextStrategy()

    class Minimal(cs.ContextStrategy):
        name = "minimal"

        def build_messages(self, agent):
            return []

    agent = FakeAgent()
    strategy = Minimal()
    assert strategy.before_turn(agent) is None
    assert strategy.after_turn(agent, "в", "о") is None
    assert agent.calls == []


# --- sliding_window -----------------------------------------------------------------------


def test_sliding_window_keeps_only_recent_pairs():
    agent = FakeAgent(_dialog(5), recent_pairs=2)
    result = cs.STRATEGIES["sliding_window"].build_messages(agent)
    assert _contents(result) == ["вопрос 4", "ответ 4", "вопрос 5", "ответ 5"]


def test_sliding_window_of_short_or_empty_dialog_returns_everything():
    assert cs.STRATEGIES["sliding_window"].build_messages(FakeAgent()) == []
    agent = FakeAgent(_dialog(1), recent_pairs=3)
    assert len(cs.STRATEGIES["sliding_window"].build_messages(agent)) == 2


def test_sliding_window_has_no_side_effects():
    agent = FakeAgent(_dialog(2))
    strategy = cs.STRATEGIES["sliding_window"]
    strategy.before_turn(agent)
    strategy.after_turn(agent, "в", "о")
    assert agent.calls == []


# --- sticky_facts -------------------------------------------------------------------------


def test_sticky_facts_puts_facts_system_message_before_recent_window():
    agent = FakeAgent(_dialog(4), facts={"горизонт": "10 лет", "риск": "низкий"}, recent_pairs=1)
    result = cs.STRATEGIES["sticky_facts"].build_messages(agent)
    assert result[0] == {
        "role": "system",
        "content": "Известные факты о диалоге:\nгоризонт: 10 лет\nриск: низкий",
    }
    assert _contents(result[1:]) == ["вопрос 4", "ответ 4"]


def test_sticky_facts_without_facts_has_only_window():
    result = cs.STRATEGIES["sticky_facts"].build_messages(FakeAgent(_dialog(3), recent_pairs=1))
    assert _contents(result) == ["вопрос 3", "ответ 3"]
    assert all(m["role"] != "system" for m in result)


def test_sticky_facts_updates_facts_before_and_after_turn():
    agent = FakeAgent()
    strategy = cs.STRATEGIES["sticky_facts"]
    strategy.before_turn(agent)
    strategy.after_turn(agent, "в", "о")
    assert agent.calls == ["facts", "facts"]


# --- branching ----------------------------------------------------------------------------


def test_branching_returns_whole_active_branch_as_a_copy():
    agent = FakeAgent(_dialog(5), recent_pairs=1)
    result = cs.STRATEGIES["branching"].build_messages(agent)
    assert result == agent._messages
    assert result is not agent._messages
    result.append({"role": "user", "content": "лишнее"})
    assert len(agent._messages) == 10


def test_branching_has_no_side_effects():
    agent = FakeAgent(_dialog(1))
    strategy = cs.STRATEGIES["branching"]
    strategy.before_turn(agent)
    strategy.after_turn(agent, "в", "о")
    assert agent.calls == []


# --- summary ------------------------------------------------------------------------------


def test_summary_adds_summary_message_and_skips_summarised_pairs():
    agent = FakeAgent(_dialog(5), summary="Обсуждали горизонт.", summarized_pairs=3)
    result = cs.STRATEGIES["summary"].build_messages(agent)
    assert result[0] == {
        "role": "system",
        "content": "Краткое содержание более ранней части этого диалога:\nОбсуждали горизонт.",
    }
    assert _contents(result[1:]) == ["вопрос 4", "ответ 4", "вопрос 5", "ответ 5"]


def test_summary_without_summary_returns_unsummarised_tail_only():
    agent = FakeAgent(_dialog(3), summary="", summarized_pairs=0)
    result = cs.STRATEGIES["summary"].build_messages(agent)
    assert _contents(result) == _contents(_dialog(3))


def test_summary_with_everything_summarised_has_only_the_summary_message():
    agent = FakeAgent(_dialog(2), summary="Всё свёрнуто.", summarized_pairs=2)
    result = cs.STRATEGIES["summary"].build_messages(agent)
    assert len(result) == 1
    assert result[0]["role"] == "system"


def test_summary_updates_summary_after_turn_only():
    agent = FakeAgent()
    strategy = cs.STRATEGIES["summary"]
    strategy.before_turn(agent)
    assert agent.calls == []
    strategy.after_turn(agent, "в", "о")
    assert agent.calls == ["summary"]
