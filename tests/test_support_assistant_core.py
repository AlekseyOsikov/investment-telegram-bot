"""Тесты матрицы исходов SupportAssistant.ask() (design.md, решение 7) на фейках: поиск, CRM и
вызов модели подставлены функциями — сети, процессов и LLM нет."""

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from llm_queue.rules import AskCancelled
from mcp_integration.market_session import ToolOutcome
from support_assistant import prompts, rules, search
from support_assistant.core import EMPTY_ANSWER_TEXT, SupportAssistant

MAIN = "ОСНОВНОЙ_ПРОМПТ"
CHAT, USER = 10, 777


@dataclass
class Chunk:
    title: str
    text: str = "текст справки"
    score: float = 0.9


def ticket(ticket_id="T-1001", status="open"):
    return {
        "ticket_id": ticket_id,
        "subject": "Нет сводок",
        "bot_feature": "/watch",
        "status": status,
        "created": "2026-10-10",
        "updated": "2026-10-14",
        "error_text": "таймаут",
        "events": [],
    }


def response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def tool_call(name, arguments="{}", call_id="c1"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments), type="function"
    )


class FakeCrm:
    def __init__(self, tickets=(), names=("get_my_profile", "list_my_tickets", "get_ticket"),
                 list_outcome=None):
        self.tool_names = list(names)
        self.openai_tools = [{"type": "function", "function": {"name": n}} for n in names]
        self.calls: list[tuple[str, str | None]] = []
        self._list = list_outcome or ToolOutcome(json.dumps({"tickets": list(tickets)}))

    async def call(self, name, raw):
        self.calls.append((name, raw))
        if name == "list_my_tickets":
            return self._list
        return ToolOutcome("{}")


class Env:
    """Собирает ассистента с фейками и запоминает, что произошло."""

    def __init__(self, retrieval, crm=None, crm_launch_error=None, replies=None,
                 model_error=None, configured=True):
        self.retrieval = retrieval
        self.crm = crm
        self.launch_error = crm_launch_error
        self.replies = list(replies if replies is not None else [response("Ответ модели.")])
        self.model_error = model_error
        self.model_calls: list[tuple[list, list | None]] = []
        self.crm_users: list[int] = []
        self.searched: list[str] = []

        def retrieve(cfg, question):
            self.searched.append(question)
            return self.retrieval

        @asynccontextmanager
        async def open_crm(user_id):
            self.crm_users.append(user_id)
            if self.launch_error:
                raise self.launch_error
            yield self.crm

        def complete(messages, tools):
            self.model_calls.append((messages, tools))
            if self.model_error:
                raise self.model_error
            return self.replies.pop(0)

        self.assistant = SupportAssistant(
            client=None,
            model="m",
            system_prompt=MAIN,
            max_output_tokens=100,
            timeout=5.0,
            search_config=search.SupportSearchConfig("idx", 5, 0.0, 2.0),
            history_turns=2,
            tickets_limit=5,
            max_tool_steps=3,
            open_crm=open_crm if configured else None,
            retrieve=retrieve,
            read_built_at=lambda d: "2026-10-17",
            complete=complete,
        )

    def ask(self, question="почему нет сводок?", should_cancel=None):
        return self.assistant.ask(CHAT, USER, question, should_cancel=should_cancel)


FOUND = search.Retrieval(search.FOUND, chunks=(Chunk("docs/support/watch.md › Сводки"),))
NOT_FOUND = search.Retrieval(search.NOT_FOUND)


def system_texts(env):
    return [m["content"] for m in env.model_calls[0][0] if m["role"] == "system"]


# --- поиск не дал опоры ----------------------------------------------------------------------


def test_no_index_is_fixed_text_without_model_or_crm():
    env = Env(search.Retrieval(search.NO_INDEX), crm=FakeCrm())
    answer = env.ask()
    assert answer.text == rules.INDEX_MISSING_TEXT and not answer.answered_by_model
    assert env.model_calls == [] and env.crm_users == []


def test_search_failed_is_fixed_text_without_model_or_crm():
    env = Env(search.Retrieval(search.FAILED, failure="x"), crm=FakeCrm())
    answer = env.ask()
    assert answer.text == rules.SEARCH_FAILED_TEXT and not answer.answered_by_model
    assert env.model_calls == [] and env.crm_users == []


def test_not_found_without_tickets_is_fixed_text():
    env = Env(NOT_FOUND, crm=FakeCrm())
    answer = env.ask()
    assert answer.text == rules.NOT_FOUND_TEXT and not answer.answered_by_model
    assert env.model_calls == [] and answer.warnings == []
    assert answer.crm_status == prompts.CRM_EMPTY


def test_not_found_with_crm_unavailable_is_fixed_text_with_warning():
    env = Env(NOT_FOUND, crm_launch_error=OSError("нет uv"))
    answer = env.ask()
    assert answer.text == rules.NOT_FOUND_TEXT and not answer.answered_by_model
    assert answer.warnings == [rules.CRM_UNAVAILABLE_WARNING]
    assert answer.crm_status == prompts.CRM_UNAVAILABLE
    assert env.model_calls == []


def test_not_found_with_tickets_calls_model_with_no_docs_note():
    env = Env(NOT_FOUND, crm=FakeCrm([ticket()]))
    answer = env.ask()
    assert answer.answered_by_model and answer.text == "Ответ модели."
    last = env.model_calls[0][0][-1]["content"]
    assert prompts.NO_DOCS_NOTE in last
    assert any('номер="T-1001"' in text for text in system_texts(env))
    assert answer.sources == ""  # фрагменты не подавались


# --- справка найдена -------------------------------------------------------------------------


def test_found_with_tickets_feeds_both_and_builds_sources_by_code():
    env = Env(FOUND, crm=FakeCrm([ticket(), ticket("T-1002", status="closed")]))
    answer = env.ask()
    messages, tools = env.model_calls[0]
    assert messages[0] == {"role": "system", "content": MAIN}
    assert messages[1]["content"] == prompts.ASSISTANT_SYSTEM_MESSAGE
    crm_text = messages[2]["content"]
    assert 'номер="T-1001"' in crm_text and "T-1002" not in crm_text  # закрытый не подаётся
    assert "три инструмента" in crm_text
    assert tools
    assert "Фрагменты справки" in messages[-1]["content"]
    assert answer.sources.startswith("Источники:")
    assert "docs/support/watch.md › Сводки" in answer.sources
    assert "2026-10-17" in answer.sources
    assert answer.crm_status == prompts.CRM_TICKETS and answer.llm_calls == 1


def test_found_without_tickets_tells_model_there_are_none():
    env = Env(FOUND, crm=FakeCrm([]))
    answer = env.ask()
    assert answer.answered_by_model and answer.crm_status == prompts.CRM_EMPTY
    assert "НЕТ открытых обращений" in system_texts(env)[2]


def test_found_with_crm_launch_failure_answers_by_docs_with_warning():
    env = Env(FOUND, crm_launch_error=OSError("нет uv"))
    answer = env.ask()
    assert answer.answered_by_model
    assert answer.warnings == [rules.CRM_UNAVAILABLE_WARNING]
    assert "НЕДОСТУПНЫ" in system_texts(env)[2]
    assert env.model_calls[0][1] is None  # инструментов у модели нет


def test_found_with_failing_list_tool_is_unavailable():
    crm = FakeCrm(list_outcome=ToolOutcome("сбой", is_error=True, transport_failure=True))
    env = Env(FOUND, crm=crm)
    answer = env.ask()
    assert answer.crm_status == prompts.CRM_UNAVAILABLE
    assert answer.warnings == [rules.CRM_UNAVAILABLE_WARNING]
    assert env.model_calls[0][1] is None


def test_crm_without_allowed_tools_is_unavailable():
    env = Env(FOUND, crm=FakeCrm(names=()))
    assert env.ask().crm_status == prompts.CRM_UNAVAILABLE


def test_crm_not_configured_answers_by_docs_only():
    env = Env(FOUND, configured=False)
    answer = env.ask()
    assert answer.answered_by_model and answer.crm_status == prompts.CRM_NOT_CONFIGURED
    assert answer.warnings == [] and env.crm_users == []
    assert "не настроена" in system_texts(env)[2]


def test_not_found_and_crm_not_configured_is_fixed_text():
    env = Env(NOT_FOUND, configured=False)
    answer = env.ask()
    assert answer.text == rules.NOT_FOUND_TEXT and env.model_calls == []


# --- изоляция и инструменты ------------------------------------------------------------------


def test_crm_opened_for_the_sender_and_tool_loop_runs():
    crm = FakeCrm([ticket()])
    calls = [tool_call("get_ticket", json.dumps({"telegram_user_id": 1, "ticket_id": "T-2001"}))]
    env = Env(FOUND, crm=crm, replies=[response(tool_calls=calls), response("Готово.")])
    # Подстановку идентификатора делает CrmTools (tests/test_support_assistant_crm.py); здесь
    # проверяется лишь, что ядро открыло CRM для отправителя и довело цикл до ответа.
    answer = env.ask()
    assert env.crm_users == [USER]
    assert answer.text == "Готово." and answer.llm_calls == 2
    assert any(name == "get_ticket" for name, _ in crm.calls)
    assert answer.tool_lines and "get_ticket" in answer.tool_lines[0]


def test_transport_failure_in_tool_adds_partial_warning():
    class FailingCrm(FakeCrm):
        async def call(self, name, raw):
            if name == "get_ticket":
                return ToolOutcome("сбой", is_error=True, transport_failure=True)
            return await super().call(name, raw)

    calls = [tool_call("get_ticket", json.dumps({"ticket_id": "T-1001"}))]
    replies = [response(tool_calls=calls), response("ок")]
    env = Env(FOUND, crm=FailingCrm([ticket()]), replies=replies)
    assert prompts.PARTIAL_DATA_WARNING in env.ask().warnings


# --- история, отмена, ошибки -----------------------------------------------------------------


def test_history_keeps_only_question_and_answer_with_limit():
    env = Env(FOUND, crm=FakeCrm([ticket()]),
              replies=[response("о1"), response("о2"), response("о3")])
    env.ask("в1"), env.ask("в2"), env.ask("в3")
    assert env.assistant.history(CHAT) == [("в2", "о2"), ("в3", "о3")]
    sent = env.model_calls[2][0]
    roles = [m["role"] for m in sent]
    assert roles[3:] == ["user", "assistant", "user", "assistant", "user"]
    assert "T-1001" not in sent[3]["content"] + sent[4]["content"]


def test_clear_empties_history():
    env = Env(FOUND, crm=FakeCrm())
    env.ask("в1")
    env.assistant.clear(CHAT)
    assert env.assistant.history(CHAT) == []


def test_fixed_text_answers_are_not_remembered():
    env = Env(NOT_FOUND, crm=FakeCrm())
    env.ask()
    assert env.assistant.history(CHAT) == []


def test_cancel_before_start_skips_everything():
    env = Env(FOUND, crm=FakeCrm())
    with pytest.raises(AskCancelled):
        env.ask(should_cancel=lambda: True)
    assert env.searched == [] and env.assistant.history(CHAT) == []


def test_cancel_after_model_call_does_not_write_history():
    state = {"n": 0}

    def should_cancel():
        state["n"] += 1
        return state["n"] > 2  # перед поиском и перед моделью — нет, перед записью — да

    env = Env(FOUND, crm=FakeCrm())
    with pytest.raises(AskCancelled):
        env.ask(should_cancel=should_cancel)
    assert env.assistant.history(CHAT) == []


def test_model_error_propagates_as_is_from_crm_path():
    env = Env(FOUND, crm=FakeCrm([ticket()]), model_error=RuntimeError("api"))
    with pytest.raises(RuntimeError, match="api"):
        env.ask()


def test_model_error_propagates_without_crm():
    env = Env(FOUND, configured=False, model_error=RuntimeError("api"))
    with pytest.raises(RuntimeError, match="api"):
        env.ask()


def test_empty_model_answer_gets_placeholder():
    env = Env(FOUND, crm=FakeCrm(), replies=[response("  ")])
    assert env.ask().text == EMPTY_ANSWER_TEXT


def test_history_turns_zero_disables_memory():
    env = Env(FOUND, crm=FakeCrm())
    env.assistant._history_turns = 0
    env.ask()
    assert env.assistant.history(CHAT) == []
