"""Ядро ассистента разработчика: вопрос → поиск по документации → (git-инструменты) → ответ с
источниками. Не импортирует Telegram: командный слой (`command.py`) — тонкая обёртка, так что
второй фасад (CLI) добавляется без переделки (design.md изменения add-dev-assistant-mode,
решения 1, 5–8).

Инварианты (не убирать):
- один поиск на вопрос по ОТДЕЛЬНОМУ индексу документации; индекс отсутствует или поиск
  сорвался — модель не вызывается;
- git-слой — `git_session.GitTools` (только ветка и список файлов, `repo_path` подставляет код);
  файловый слой — `files_session.FilesTools` (чтение, поиск, предложение правки; применение и
  отклонение — только по кнопке, `decide_proposal`); недоступность любого сервера не блокирует
  ответ по документации и остальные слои;
- содержимое файлов и предложений в историю режима не попадает (в `_remember` идёт итог ответа);
- источники строит код по фактически поданным фрагментам (`prompts.format_sources`);
- история режима — только в памяти процесса: итог «вопрос — ответ» без фрагментов и вызовов
  инструментов; на диск не пишется; пишется ПОСЛЕ проверки отмены;
- `SYSTEM_PROMPT` основного потока уходит первым сообщением без изменений.

`ask()` синхронен и выполняется в потоке под слотом очереди (путь с серверами — `asyncio.run`, как
у SmartAgent); исключения основного вызова модели доходят до командного слоя КАК ЕСТЬ."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agents import market_tools
from llm_queue.rules import raise_if_cancelled

from . import indexer, prompts, proposals, rules, search
from .files_session import CombinedTools, FilesSettings, open_files_tools, run_button_action
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
    proposals: list[proposals.Proposal] = field(default_factory=list)


@dataclass
class _ToolRun:
    """Итог работы с серверами слоёв на один вопрос."""

    result: market_tools.ToolLoopResult | None = None
    git_available: bool = False
    files_available: bool = False
    proposals: list[proposals.Proposal] = field(default_factory=list)
    done: bool = False


# Исходы нажатия кнопки под предложением.
OUTCOME_STALE = "stale"
OUTCOME_FOREIGN = "foreign"
OUTCOME_APPLIED = "applied"
OUTCOME_DISCARDED = "discarded"
OUTCOME_FAILED = "failed"


@dataclass(frozen=True)
class ProposalOutcome:
    """Итог нажатия кнопки: `FOREIGN` — молча проигнорировать (чужое предложение)."""

    status: str
    text: str = ""


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
        files: FilesSettings | None = None,
        patches_dir: str = "",
        registry: proposals.ProposalRegistry | None = None,
        open_files: Callable[..., Any] = open_files_tools,
        run_button: Callable[..., Any] = run_button_action,
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
        self._files = files
        self._patches_dir = patches_dir
        self._registry = registry or proposals.ProposalRegistry(
            files.ttl_seconds if files is not None else 3600
        )
        self._open_files = open_files
        self._run_button = run_button
        self._history: dict[int, deque[tuple[str, str]]] = {}

    # --- история режима (только память процесса) -----------------------------------------

    def history(self, chat_id: int) -> list[tuple[str, str]]:
        return list(self._history.get(chat_id, ()))

    def clear(self, chat_id: int) -> None:
        """Очищает краткосрочный контекст чата (выход из режима); ожидающие кнопки чата
        перестают действовать (файлы предложений на сервере дожидаются срока жизни)."""
        self._history.pop(chat_id, None)
        self._registry.clear_chat(chat_id)

    def _remember(self, chat_id: int, question: str, answer: str) -> None:
        if self._history_turns <= 0:
            return
        turns = self._history.setdefault(chat_id, deque(maxlen=self._history_turns))
        turns.append((question, answer))

    # --- вызов модели ---------------------------------------------------------------------

    @property
    def git_configured(self) -> bool:
        return bool(self._git_directory)

    @property
    def files_configured(self) -> bool:
        return self._files is not None

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

    async def _enter(self, stack: AsyncExitStack, manager: Any, label: str) -> Any | None:
        """Открывает сессию слоя в общем стеке; сбой запуска — `None` (слой недоступен), остальные
        слои и ответ по документации не страдают."""
        try:
            return await stack.enter_async_context(manager)
        except Exception:  # noqa: BLE001 — сбой запуска одного слоя не блокирует остальные
            logger.warning("Не удалось запустить %s.", label, exc_info=True)
            return None

    async def _with_tools(
        self,
        build: Callable[[str, str | None], list[dict]],
        retrieval: search.DevRetrieval,
        should_cancel: Callable[[], bool] | None,
        chat_id: int,
        user_id: int,
    ) -> _ToolRun:
        """Открывает серверы слоёв (git, файлы) на ЭТОТ вопрос и ведёт общий цикл инструментов.
        `result=None` — инструментов нет, а подавать модели нечего (фиксированный отказ).

        Ошибка основного вызова модели ловится ВНУТРИ сессий и возбуждается заново после выхода:
        anyio может завернуть её в ExceptionGroup, обойдя `except` командного слоя. Сбой закрытия
        сессий при готовом ответе ответ не отменяет — только журнал."""
        run = _ToolRun()
        body_error: Exception | None = None
        try:
            async with AsyncExitStack() as stack:
                git = files = None
                if self.git_configured:
                    git = await self._enter(
                        stack,
                        self._open_git(
                            self._git_directory,
                            self._git_root,
                            self._mcp_timeout,
                            self._tool_result_max_chars,
                        ),
                        "mcp-git",
                    )
                if self.files_configured:
                    files = await self._enter(
                        stack,
                        self._open_files(
                            self._files,
                            self._mcp_timeout,
                            self._tool_result_max_chars,
                            self._registry,
                            chat_id,
                            user_id,
                        ),
                        "файловый сервер",
                    )
                run.git_available = git is not None and bool(git.openai_tools)
                run.files_available = files is not None and bool(files.openai_tools)
                layers = [
                    layer
                    for layer, ok in ((git, run.git_available), (files, run.files_available))
                    if ok
                ]
                run.done = True
                if not layers and retrieval.outcome == search.NOT_FOUND:
                    return run
                try:
                    git_status = (
                        prompts.GIT_AVAILABLE
                        if run.git_available
                        else prompts.GIT_UNAVAILABLE
                        if self.git_configured
                        else prompts.GIT_NOT_CONFIGURED
                    )
                    files_status = (
                        None
                        if not self.files_configured
                        else prompts.FILES_AVAILABLE
                        if run.files_available
                        else prompts.FILES_UNAVAILABLE
                    )
                    messages = build(git_status, files_status)
                    if layers:
                        tools = CombinedTools(*layers)

                        def complete(history: list[dict], with_tools: bool):
                            return self._chat(
                                history, tools.openai_tools if with_tools else None, should_cancel
                            )

                        run.result = await market_tools.run_tool_loop(
                            messages, complete, tools.call, self._max_tool_steps
                        )
                    else:
                        run.result = self._plain(messages, should_cancel)
                    if files is not None:
                        run.proposals = list(files.proposals)
                except Exception as exc:  # noqa: BLE001 — возбуждается ниже, см. докстринг
                    body_error = exc
        except Exception:  # noqa: BLE001
            logger.warning("Сбой при завершении сессий MCP-серверов.", exc_info=True)
            if not run.done:
                # Сессии не удалось даже открыть: отвечаем без инструментов.
                if retrieval.outcome == search.NOT_FOUND:
                    run.done = True
                    return run
                git_status = (
                    prompts.GIT_UNAVAILABLE if self.git_configured else prompts.GIT_NOT_CONFIGURED
                )
                files_status = prompts.FILES_UNAVAILABLE if self.files_configured else None
                run.result = self._plain(build(git_status, files_status), should_cancel)
        if body_error is not None:
            raise body_error
        return run

    # --- вопрос ---------------------------------------------------------------------------

    def ask(
        self,
        chat_id: int,
        question: str,
        should_cancel: Callable[[], bool] | None = None,
        user_id: int | None = None,
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

        def build(git_status: str, files_status: str | None = None) -> list[dict]:
            return prompts.build_messages(
                self._system_prompt,
                git_status,
                history,
                question,
                chunks,
                no_docs=no_docs,
                files_status=files_status,
            )

        warnings: list[str] = []
        new_proposals: list[proposals.Proposal] = []
        if self.git_configured or self.files_configured:
            owner = chat_id if user_id is None else user_id
            run = asyncio.run(self._with_tools(build, retrieval, should_cancel, chat_id, owner))
            result, new_proposals = run.result, run.proposals
            if self.git_configured and not run.git_available:
                warnings.append(rules.GIT_UNAVAILABLE_WARNING)
            if self.files_configured and not run.files_available:
                warnings.append(rules.FILES_UNAVAILABLE_WARNING)
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
            proposals=new_proposals,
        )

    # --- кнопки под предложениями ---------------------------------------------------------

    def _save_patch(self, proposal_id: str, diff: str) -> str | None:
        """Сохраняет применённый diff как `.patch` (`git apply`); возвращает полный путь к файлу,
        `None` — сохранить не удалось."""
        if not self._patches_dir or not diff:
            return None
        name = proposals.patch_filename(proposal_id)
        try:
            directory = Path(self._patches_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text(diff, encoding="utf-8")
        except OSError:
            logger.warning("Не удалось сохранить патч %s.", name, exc_info=True)
            return None
        return str(directory / name)

    async def decide_proposal(
        self, action: str, proposal_id: str, chat_id: int, user_id: int
    ) -> ProposalOutcome:
        """Нажатие «Применить»/«Отклонить». Запись выполняет сервер по этому вызову кода, а не
        модель. Предложение вынимается из реестра ДО обращения к серверу (двойное нажатие
        получает «устарело»); чужое — `FOREIGN` без изменений."""
        if not self.files_configured or action not in (
            proposals.ACTION_APPLY,
            proposals.ACTION_DISCARD,
        ):
            return ProposalOutcome(OUTCOME_STALE, proposals.STALE_TEXT)
        record = self._registry.get(proposal_id)
        if record is None:
            return ProposalOutcome(OUTCOME_STALE, proposals.STALE_TEXT)
        if not proposals.is_owner(record, chat_id, user_id):
            return ProposalOutcome(OUTCOME_FOREIGN)
        self._registry.pop(proposal_id)
        apply = action == proposals.ACTION_APPLY
        try:
            outcome = await self._run_button(
                "apply_change" if apply else "discard_change",
                proposal_id,
                self._files,
                self._mcp_timeout,
            )
        except Exception:  # noqa: BLE001 — сервер не запустился: файл не менялся
            logger.warning("Не удалось запустить файловый сервер для кнопки.", exc_info=True)
            return ProposalOutcome(OUTCOME_FAILED, proposals.SERVER_FAILURE_TEXT)
        if outcome.transport_failure:
            return ProposalOutcome(OUTCOME_FAILED, proposals.UNKNOWN_STATE_TEXT)
        if not apply:
            return ProposalOutcome(OUTCOME_DISCARDED, proposals.DISCARDED_TEXT)
        if outcome.is_error:
            message = proposals.tool_error_message(outcome.text, "apply_change")
            return ProposalOutcome(OUTCOME_FAILED, proposals.failure_text(message))
        try:
            data = json.loads(outcome.text)
            path, added, removed = str(data["path"]), int(data["added"]), int(data["removed"])
            diff = str(data.get("diff", ""))
        except (ValueError, KeyError, TypeError):
            return ProposalOutcome(OUTCOME_FAILED, proposals.UNKNOWN_STATE_TEXT)
        patch_name = await asyncio.to_thread(self._save_patch, proposal_id, diff)
        return ProposalOutcome(
            OUTCOME_APPLIED, proposals.applied_text(path, added, removed, patch_name)
        )
