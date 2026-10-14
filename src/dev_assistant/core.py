"""Ядро ассистента разработчика: вопрос → поиск по документации → (git-инструменты) → ответ с
источниками. Не импортирует Telegram: командный слой (`command.py`) — тонкая обёртка, так что
второй фасад (CLI) добавляется без переделки (design.md изменения add-dev-assistant-mode,
решения 1, 5–8).

Инварианты (не убирать):
- один поиск на вопрос по ОТДЕЛЬНОМУ индексу документации; индекс отсутствует или поиск
  сорвался — модель не вызывается;
- git-слой — `git_session.GitTools` (только ветка и список файлов, `repo_path` подставляет код);
  недоступность сервера не блокирует ответ по документации;
- источники строит код по фактически поданным фрагментам (`prompts.format_sources`);
- история режима — только в памяти процесса: итог «вопрос — ответ» без фрагментов и вызовов
  инструментов; на диск не пишется; пишется ПОСЛЕ проверки отмены;
- `SYSTEM_PROMPT` основного потока уходит первым сообщением без изменений.

`ask()` синхронен и выполняется в потоке под слотом очереди (путь с git — `asyncio.run`, как у
SmartAgent); исключения основного вызова модели доходят до командного слоя КАК ЕСТЬ."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from agents import market_tools
from llm_queue.rules import raise_if_cancelled

from . import indexer, prompts, rules, search
from .git_session import display_arguments, open_git_tools

logger = logging.getLogger(__name__)

EMPTY_ANSWER_TEXT = "Модель вернула пустой ответ. Попробуй переформулировать вопрос."


@dataclass
class DevAnswer:
    """Итог одного вопроса. `answered_by_model=False` — отказ или служебный ответ кода без
    вызова модели (индекс не собран, поиск сорвался, нечего подать)."""

    text: str
    sources: str = ""
    tool_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    answered_by_model: bool = True
    llm_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class DevAssistant:
    """Ассистент разработчика. Все настройки — параметры конструктора (значения берёт
    `command.py` из `config`): ядро не читает окружение."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        system_prompt: str,
        max_output_tokens: int,
        timeout: float,
        search_config: search.DevSearchConfig,
        history_turns: int,
        git_directory: str,
        git_root: str,
        mcp_timeout: float,
        max_tool_steps: int,
        tool_result_max_chars: int,
        retrieve: Callable[..., search.DevRetrieval] = search.retrieve,
        read_built_at: Callable[[str], str | None] = indexer.read_built_at,
        open_git: Callable[..., Any] = open_git_tools,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._search_config = search_config
        self._history_turns = max(history_turns, 0)
        self._git_directory = git_directory
        self._git_root = git_root
        self._mcp_timeout = mcp_timeout
        self._max_tool_steps = max_tool_steps
        self._tool_result_max_chars = tool_result_max_chars
        self._retrieve = retrieve
        self._read_built_at = read_built_at
        self._open_git = open_git
        self._history: dict[int, deque[tuple[str, str]]] = {}

    # --- история режима (только память процесса) -----------------------------------------

    def history(self, chat_id: int) -> list[tuple[str, str]]:
        return list(self._history.get(chat_id, ()))

    def clear(self, chat_id: int) -> None:
        """Очищает краткосрочный контекст чата (выход из режима)."""
        self._history.pop(chat_id, None)

    def _remember(self, chat_id: int, question: str, answer: str) -> None:
        if self._history_turns <= 0:
            return
        turns = self._history.setdefault(chat_id, deque(maxlen=self._history_turns))
        turns.append((question, answer))

    # --- вызов модели ---------------------------------------------------------------------

    @property
    def git_configured(self) -> bool:
        return bool(self._git_directory)

    def _chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        should_cancel: Callable[[], bool] | None,
    ):
        # Точка кооперативной отмены перед КАЖДЫМ обращением к модели (в т.ч. в цикле tools).
        raise_if_cancelled(should_cancel)
        kwargs = {"tools": tools} if tools else {}
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_output_tokens,
            timeout=self._timeout,
            **kwargs,
        )

    def _plain(
        self, messages: list[dict], should_cancel: Callable[[], bool] | None
    ) -> market_tools.ToolLoopResult:
        response = self._chat(messages, None, should_cancel)
        usage = getattr(response, "usage", None)
        return market_tools.ToolLoopResult(
            text=response.choices[0].message.content,
            llm_calls=1,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    async def _with_git(
        self,
        build: Callable[[str], list[dict]],
        retrieval: search.DevRetrieval,
        should_cancel: Callable[[], bool] | None,
    ) -> tuple[market_tools.ToolLoopResult | None, bool]:
        """Открывает `mcp-git` на ЭТОТ вопрос и ведёт цикл инструментов. Возвращает
        (результат, git_доступен). `None` результата — git недоступен, а подавать модели
        нечего (вызывающий код отвечает фиксированным текстом).

        Ошибка основного вызова модели ловится ВНУТРИ сессии и возбуждается заново после
        выхода: anyio может завернуть её в ExceptionGroup, обойдя `except` командного слоя.
        Сбой закрытия сессии при готовом ответе ответ не отменяет — только журнал."""
        result: market_tools.ToolLoopResult | None = None
        available = False
        body_error: Exception | None = None
        try:
            async with self._open_git(
                self._git_directory,
                self._git_root,
                self._mcp_timeout,
                self._tool_result_max_chars,
            ) as tools:
                available = bool(tools.openai_tools)
                try:
                    status = prompts.GIT_AVAILABLE if available else prompts.GIT_UNAVAILABLE
                    if not available and retrieval.outcome == search.NOT_FOUND:
                        return None, False
                    messages = build(status)
                    if available:

                        def complete(history: list[dict], with_tools: bool):
                            return self._chat(
                                history, tools.openai_tools if with_tools else None, should_cancel
                            )

                        result = await market_tools.run_tool_loop(
                            messages, complete, tools.call, self._max_tool_steps
                        )
                    else:
                        result = self._plain(messages, should_cancel)
                except Exception as exc:  # noqa: BLE001 — возбуждается ниже, см. докстринг
                    body_error = exc
        except Exception:  # noqa: BLE001
            if body_error is None and result is None:
                # Сервер не запустился (запуск — исключение open_market_tools).
                logger.warning("Не удалось запустить mcp-git.", exc_info=True)
                return await self._without_git(build, retrieval, should_cancel)
            logger.warning("Сбой при завершении сессии mcp-git.", exc_info=True)
        if body_error is not None:
            raise body_error
        return result, available

    async def _without_git(
        self,
        build: Callable[[str], list[dict]],
        retrieval: search.DevRetrieval,
        should_cancel: Callable[[], bool] | None,
    ) -> tuple[market_tools.ToolLoopResult | None, bool]:
        if retrieval.outcome == search.NOT_FOUND:
            return None, False
        return self._plain(build(prompts.GIT_UNAVAILABLE), should_cancel), False

    # --- вопрос ---------------------------------------------------------------------------

    def ask(
        self,
        chat_id: int,
        question: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> DevAnswer:
        raise_if_cancelled(should_cancel)
        retrieval = self._retrieve(self._search_config, question)

        if retrieval.outcome == search.NO_INDEX:
            return DevAnswer(rules.INDEX_MISSING_TEXT, answered_by_model=False)
        if retrieval.outcome == search.FAILED:
            return DevAnswer(rules.SEARCH_FAILED_TEXT, answered_by_model=False)

        chunks = retrieval.chunks
        no_docs = retrieval.outcome == search.NOT_FOUND
        history = self.history(chat_id)

        def build(git_status: str) -> list[dict]:
            return prompts.build_messages(
                self._system_prompt, git_status, history, question, chunks, no_docs=no_docs
            )

        warnings: list[str] = []
        if self.git_configured:
            result, git_available = asyncio.run(self._with_git(build, retrieval, should_cancel))
            if not git_available:
                warnings.append(rules.GIT_UNAVAILABLE_WARNING)
        elif no_docs:
            result = None  # нечего подать и нет инструментов: фиксированный отказ без модели
        else:
            result = self._plain(build(prompts.GIT_NOT_CONFIGURED), should_cancel)

        if result is None:
            return DevAnswer(rules.NOT_FOUND_TEXT, warnings=warnings, answered_by_model=False)

        text = (result.text or "").strip() or EMPTY_ANSWER_TEXT
        if result.transport_failure:
            warnings.append(prompts.GIT_PARTIAL_WARNING)
        if result.step_limit_reached:
            warnings.append(market_tools.step_limit_warning(self._max_tool_steps))

        # Отмена проверяется ПЕРЕД записью в память: снятый вопрос в контекст не попадает.
        raise_if_cancelled(should_cancel)
        self._remember(chat_id, question, text)

        return DevAnswer(
            text=text,
            sources=prompts.format_sources(
                chunks, self._read_built_at(self._search_config.index_dir)
            ),
            tool_lines=market_tools.format_call_lines(
                [
                    replace(call, arguments=display_arguments(call.arguments))
                    for call in result.calls
                ]
            ),
            warnings=warnings,
            llm_calls=result.llm_calls,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
