"""Ядро ассистента поддержки: вопрос → поиск по справке → (CRM: обращения пользователя) → ответ с
источниками. Не импортирует Telegram: командный слой (`command.py`) — тонкая обёртка (design.md
изменения add-support-assistant, решения 2, 3, 7, 8).

Инварианты (не убирать):
- один поиск на вопрос по ОТДЕЛЬНОМУ индексу справки; индекса нет или поиск сорвался — модель не
  вызывается;
- обращения пользователя запрашивает КОД на каждый вопрос (`crm_session.preload_tickets`), а не
  модель; идентификатор пользователя подставляет `CrmTools` — модель его не выбирает;
- недоступность CRM не блокирует ответ по справке (модель получает «НЕДОСТУПНЫ», пользователь —
  предупреждение); если и справка ничего не нашла, а обращений для подачи нет — фиксированный
  текст БЕЗ вызова модели (матрица исходов — design.md, решение 7);
- источники строит код по фактически поданным фрагментам (`prompts.format_sources`);
- история режима — только в памяти процесса: итог «вопрос — ответ» без фрагментов, данных
  обращений и вызовов инструментов; на диск не пишется; пишется ПОСЛЕ проверки отмены;
- `SYSTEM_PROMPT` основного потока уходит первым сообщением без изменений.

`ask()` синхронен и выполняется в потоке под слотом очереди (путь с CRM — `asyncio.run`, как у
`/dev`); исключения основного вызова модели доходят до командного слоя КАК ЕСТЬ."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from agents import market_tools
from llm_queue.rules import raise_if_cancelled

from . import crm_session, indexer, prompts, rules, search

logger = logging.getLogger(__name__)

EMPTY_ANSWER_TEXT = "Модель вернула пустой ответ. Попробуй переформулировать вопрос."


@dataclass
class SupportAnswer:
    """Итог одного вопроса. `answered_by_model=False` — отказ или служебный ответ кода без
    вызова модели (индекс не собран, поиск сорван, нечего подать). `crm_status` — `CRM_*` из
    `prompts` (что узнал код об обращениях)."""

    text: str
    sources: str = ""
    tool_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    answered_by_model: bool = True
    crm_status: str = prompts.CRM_NOT_CONFIGURED
    llm_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class _CrmRun:
    """Итог работы с CRM на один вопрос: ответ модели (None — модель не вызывалась) и состояние
    данных обращений."""

    result: market_tools.ToolLoopResult | None
    status: str


class SupportAssistant:
    """Ассистент поддержки. Все настройки — параметры конструктора (значения берёт `command.py`
    из `config`): ядро не читает окружение. `open_crm(user_id)` — фабрика асинхронного контекста
    с `CrmTools` (None — CRM не настроена); `complete(messages, tools)` — один обычный вызов модели
    (по умолчанию — `client.chat.completions.create`)."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        system_prompt: str,
        max_output_tokens: int,
        timeout: float,
        search_config: search.SupportSearchConfig,
        history_turns: int,
        tickets_limit: int,
        max_tool_steps: int,
        open_crm: Callable[[int], Any] | None = None,
        retrieve: Callable[..., search.Retrieval] = search.retrieve,
        read_built_at: Callable[[str], str | None] = indexer.read_built_at,
        complete: Callable[[list[dict], list[dict] | None], Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._system_prompt = system_prompt
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._search_config = search_config
        self._history_turns = max(history_turns, 0)
        self._tickets_limit = tickets_limit
        self._max_tool_steps = max_tool_steps
        self._open_crm = open_crm
        self._retrieve = retrieve
        self._read_built_at = read_built_at
        self._complete = complete or self._default_complete
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
    def crm_configured(self) -> bool:
        return self._open_crm is not None

    def _default_complete(self, messages: list[dict], tools: list[dict] | None) -> Any:
        kwargs = {"tools": tools} if tools else {}
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_output_tokens,
            timeout=self._timeout,
            **kwargs,
        )

    def _chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> Any:
        # Точка кооперативной отмены перед КАЖДЫМ обращением к модели (в т.ч. в цикле tools).
        raise_if_cancelled(should_cancel)
        return self._complete(messages, tools)

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

    async def _with_crm(
        self,
        user_id: int,
        no_docs: bool,
        build: Callable[[dict[str, str]], list[dict]],
        should_cancel: Callable[[], bool] | None,
    ) -> _CrmRun:
        """Открывает CRM на ЭТОТ вопрос, предзагружает обращения и ведёт цикл инструментов.

        Ошибка основного вызова модели ловится ВНУТРИ сессии и возбуждается заново после выхода:
        anyio может завернуть её в ExceptionGroup, обойдя `except` командного слоя. Сбой
        закрытия сессии при готовом результате его не отменяет — только журнал."""
        run: _CrmRun | None = None
        entered = False
        body_error: Exception | None = None
        try:
            async with self._open_crm(user_id) as crm:  # type: ignore[misc]
                entered = True
                try:
                    if crm.openai_tools:
                        snapshot = await crm_session.preload_tickets(crm, self._tickets_limit)
                    else:
                        snapshot = crm_session.TicketsSnapshot(
                            crm_session.TICKETS_UNAVAILABLE, failure="нет разрешённых инструментов"
                        )
                    if snapshot.status != crm_session.TICKETS_OK:
                        status = prompts.CRM_UNAVAILABLE
                    elif snapshot.tickets:
                        status = prompts.CRM_TICKETS
                    else:
                        status = prompts.CRM_EMPTY
                    if no_docs and status != prompts.CRM_TICKETS:
                        run = _CrmRun(None, status)  # нечего подать: фиксированный текст
                    else:
                        with_tools = snapshot.status == crm_session.TICKETS_OK
                        messages = build(
                            prompts.crm_layer_message(
                                status, snapshot.tickets, snapshot.omitted, with_tools=with_tools
                            )
                        )
                        if with_tools:

                            def complete(history: list[dict], use_tools: bool) -> Any:
                                return self._chat(
                                    history, crm.openai_tools if use_tools else None, should_cancel
                                )

                            result = await market_tools.run_tool_loop(
                                messages, complete, crm.call, self._max_tool_steps
                            )
                        else:
                            result = self._plain(messages, should_cancel)
                        run = _CrmRun(result, status)
                except Exception as exc:  # noqa: BLE001 — возбуждается ниже, см. докстринг
                    body_error = exc
        except Exception:  # noqa: BLE001
            if not entered:
                # Сервер не запустился (запуск — исключение open_market_tools).
                logger.warning("Не удалось запустить CRM-сервер поддержки.", exc_info=True)
                return self._without_crm(prompts.CRM_UNAVAILABLE, no_docs, build, should_cancel)
            logger.warning("Сбой при завершении сессии CRM-сервера поддержки.", exc_info=True)
        if body_error is not None:
            raise body_error
        if run is None:  # сессия вошла, но ни ответа, ни ошибки (не ожидается)
            return self._without_crm(prompts.CRM_UNAVAILABLE, no_docs, build, should_cancel)
        return run

    def _without_crm(
        self,
        status: str,
        no_docs: bool,
        build: Callable[[dict[str, str]], list[dict]],
        should_cancel: Callable[[], bool] | None,
    ) -> _CrmRun:
        """Ответ без данных обращений: CRM недоступна или не настроена."""
        if no_docs:
            return _CrmRun(None, status)
        messages = build(prompts.crm_layer_message(status))
        return _CrmRun(self._plain(messages, should_cancel), status)

    # --- вопрос ---------------------------------------------------------------------------

    def ask(
        self,
        chat_id: int,
        user_id: int,
        question: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> SupportAnswer:
        raise_if_cancelled(should_cancel)
        retrieval = self._retrieve(self._search_config, question)

        if retrieval.outcome == search.NO_INDEX:
            return SupportAnswer(rules.INDEX_MISSING_TEXT, answered_by_model=False)
        if retrieval.outcome == search.FAILED:
            return SupportAnswer(rules.SEARCH_FAILED_TEXT, answered_by_model=False)

        chunks = retrieval.chunks
        no_docs = retrieval.outcome == search.NOT_FOUND
        history = self.history(chat_id)

        def build(crm_message: dict[str, str]) -> list[dict]:
            return prompts.build_messages(
                self._system_prompt, crm_message, history, question, chunks, no_docs=no_docs
            )

        if self.crm_configured:
            run = asyncio.run(self._with_crm(user_id, no_docs, build, should_cancel))
        else:
            run = self._without_crm(prompts.CRM_NOT_CONFIGURED, no_docs, build, should_cancel)

        warnings: list[str] = []
        if run.status == prompts.CRM_UNAVAILABLE:
            warnings.append(rules.CRM_UNAVAILABLE_WARNING)

        result = run.result
        if result is None:
            return SupportAnswer(
                rules.NOT_FOUND_TEXT,
                warnings=warnings,
                answered_by_model=False,
                crm_status=run.status,
            )

        text = (result.text or "").strip() or EMPTY_ANSWER_TEXT
        if result.transport_failure:
            warnings.append(prompts.PARTIAL_DATA_WARNING)
        if result.step_limit_reached:
            warnings.append(market_tools.step_limit_warning(self._max_tool_steps))

        # Отмена проверяется ПЕРЕД записью в память: снятый вопрос в контекст не попадает.
        raise_if_cancelled(should_cancel)
        self._remember(chat_id, question, text)

        return SupportAnswer(
            text=text,
            sources=prompts.format_sources(
                chunks, self._read_built_at(self._search_config.index_dir)
            ),
            tool_lines=market_tools.format_call_lines(
                [
                    replace(call, arguments=crm_session.display_arguments(call.arguments))
                    for call in result.calls
                ]
            ),
            warnings=warnings,
            crm_status=run.status,
            llm_calls=result.llm_calls,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
