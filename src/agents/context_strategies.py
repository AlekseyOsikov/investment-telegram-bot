"""Стратегии управления контекстом `/agent` — вынесены из `Agent` в отдельные классы
с общим интерфейсом (см. докстринг модуля `agents/agent.py`, раздел "Управление
контекстом", про то, что значит каждая из 4 стратегий на уровне поведения; здесь —
только их код).

Каждая стратегия отвечает на три вопроса:
1. before_turn(agent) — что сделать ДО сборки контекста и отправки вопроса в LLM (по
   умолчанию — ничего; sticky_facts использует это, чтобы догнать бэклог ещё не
   учтённых в facts пар с прошлых ходов до того, как контекст для текущего вопроса
   будет собран, — см. StickyFactsStrategy).
2. build_messages(agent) — какие сообщения поставить между системным промптом и
   вопросом пользователя (сам системный промпт и сам вопрос — общий код в
   Agent._build_context_messages()/Agent.ask(), не часть стратегии).
3. after_turn(agent, user_text, answer_text) — что сделать после того, как пара
   вопрос-ответ уже дописана в активную ветку (для sliding_window/branching делать
   нечего; sticky_facts и summary оба что-то делают здесь, каждая по своей причине —
   см. их докстринги).

Стратегии тесно связаны с Agent и читают/пишут его "protected" (однократное
подчёркивание) состояние напрямую — active-ветку, facts, summary — вместо
дублирования этого состояния у себя: персистентность (JSON-файл на диске) и вызовы
LLM-клиента остаются полностью на стороне Agent (self._client/self._model/
self._timeout, self._save_history()), стратегии только решают, что именно
происходит с уже загруженным состоянием. Это осознанный компромисс ради простоты —
не общий Protocol с изолированным состоянием на каждую стратегию, а тонкий
диспетчер поведения поверх одного и того же Agent.

STRATEGIES — реестр по имени, используемый Agent.ask() (before_turn),
Agent._build_context_messages() (build_messages) и Agent._after_turn() (after_turn).
Набор имён-ключей должен совпадать с AGENT_STRATEGY_LABELS (config.py) — это две
стороны одного и того же списка стратегий (человекочитаемые подписи и поведение),
проверяется в Agent._load_state()/config._validate_config() отдельно.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


class ContextStrategy(ABC):
    """Общий интерфейс стратегии управления контекстом (см. докстринг модуля)."""

    name: str

    def before_turn(self, agent: Agent) -> None:  # noqa: B027 — намеренно
        # конкретный no-op, а не abstractmethod: большинство стратегий его не
        # переопределяют.
        """Побочное действие ДО сборки контекста и отправки вопроса в LLM. По
        умолчанию — ничего (переопределяет только sticky_facts)."""

    @abstractmethod
    def build_messages(self, agent: Agent) -> list[dict[str, str]]:
        """Сообщения между системным промптом и вопросом пользователя."""

    def after_turn(  # noqa: B027 — намеренно конкретный no-op, а не abstractmethod:
        # sliding_window/branching этот метод не переопределяют.
        self, agent: Agent, user_text: str, answer_text: str
    ) -> None:
        """Побочное действие после успешно сохранённой пары вопрос-ответ. По
        умолчанию — ничего (переопределяют sticky_facts и summary)."""


class SlidingWindowStrategy(ContextStrategy):
    """Последние agent._recent_pairs пар активной ветки, всё остальное отбрасывается
    из контекста (не из хранения на диске) — см. "Sliding Window" в докстринге
    agents/agent.py."""

    name = "sliding_window"

    def build_messages(self, agent: Agent) -> list[dict[str, str]]:
        return agent._recent_window(agent._active_messages())


class StickyFactsStrategy(ContextStrategy):
    """Словарь фактов (agent._facts) одним системным сообщением + последние
    agent._recent_pairs пар — см. "Sticky Facts" в докстринге agents/agent.py.

    Факты обновляются вызовом LLM (agent._update_facts()) ДВАЖДЫ вокруг каждого
    вопроса, чтобы facts всегда были максимально актуальны:
    - before_turn — ДО сборки контекста и отправки текущего вопроса: догоняет
      бэклог ещё не учтённых пар с ПРЕДЫДУЩИХ ходов (включая пропущенные из-за
      прошлых сбоев), если он есть, — так контекст самого текущего вопроса
      строится уже на актуальных facts, а не на устаревших.
    - after_turn — сразу ПОСЛЕ получения ответа на текущий вопрос (пара уже
      дописана в активную ветку к этому моменту): учитывает саму эту пару, не
      дожидаясь следующего вопроса.
    _update_facts() сам берёт бэклог необработанных пар из agent._active_messages()
    (см. его докстринг) и ничего не делает, если бэклог пуст, — поэтому оба вызова
    безопасны: before_turn обычно не видит ничего нового (если предыдущий after_turn
    уже всё учёл), а after_turn обрабатывает ровно одну свежую пару. Цена такой
    актуальности — до двух дополнительных вызовов LLM на вопрос вместо одного (когда
    есть что обновлять), а не один, как раньше."""

    name = "sticky_facts"

    def before_turn(self, agent: Agent) -> None:
        agent._update_facts()

    def after_turn(self, agent: Agent, user_text: str, answer_text: str) -> None:
        agent._update_facts()

    def build_messages(self, agent: Agent) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if agent._facts:
            facts_text = "\n".join(f"{key}: {value}" for key, value in agent._facts.items())
            messages.append(
                {"role": "system", "content": f"Известные факты о диалоге:\n{facts_text}"}
            )
        messages.extend(agent._recent_window(agent._active_messages()))
        return messages


class BranchingStrategy(ContextStrategy):
    """Вся история активной ветки целиком, без обрезки и без summary/facts — суть
    стратегии в независимом продолжении диалога в каждой ветке, а не в экономии
    контекста, см. "Branching" в докстринге agents/agent.py. Сами операции с
    чекпоинтами/ветками (create_checkpoint/create_branch/switch_branch) живут в
    Agent, а не здесь — это управление ХРАНЕНИЕМ (общим для всех стратегий), а не
    сборка контекста для конкретного запроса."""

    name = "branching"

    def build_messages(self, agent: Agent) -> list[dict[str, str]]:
        return list(agent._active_messages())


class SummaryStrategy(ContextStrategy):
    """Summary (сводка более ранней части) + несвёрнутый хвост активной ветки — см.
    "summary" в докстринге agents/agent.py. Стратегия по умолчанию, в т.ч. для чатов
    с JSON-файлом истории ещё в старом (доветочном) формате."""

    name = "summary"

    def build_messages(self, agent: Agent) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if agent._summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Краткое содержание более ранней части этого диалога:\n{agent._summary}"
                    ),
                }
            )
        messages.extend(agent._active_messages()[2 * agent._summarized_pairs :])
        return messages

    def after_turn(self, agent: Agent, user_text: str, answer_text: str) -> None:
        agent._maybe_update_summary()


STRATEGIES: dict[str, ContextStrategy] = {
    strategy.name: strategy
    for strategy in (
        SlidingWindowStrategy(),
        StickyFactsStrategy(),
        BranchingStrategy(),
        SummaryStrategy(),
    )
}
