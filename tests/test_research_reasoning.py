"""Тесты чистых частей research/reasoning.py (/research_reasoning): подстановка рассуждений
вместо пустого ответа, форматирование ответа, клавиатуры, перевод ошибок и статистика сценария,
состав ролей экспертов и подзадач сравнения.

Без сети, Telegram и модели; обработчик сценария подставляется простой функцией. Не покрыто
(нужны модель, очередь или Telegram): call_reasoning_direct, call_reasoning_step_by_step,
call_reasoning_self_prompt, call_reasoning_expert, call_reasoning_compare_all,
_send_scenario_result, reasoning_command, reasoning_receive_task, reasoning_scenario_callback,
reasoning_expert_callback, build_reasoning_conversation_handler.
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError, RateLimitError

from research import reasoning as r


def _message(content, reasoning=None):
    return SimpleNamespace(content=content, reasoning_content=reasoning)


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://example.invalid/v1")


def _failing(exc):
    def handler(task, extra):
        raise exc

    return handler


# --- content / reasoning ---------------------------------------------------------------


def test_visible_content_is_returned_unchanged():
    assert r._content_or_reasoning_fallback(_message("Итог", "мысли")) == "Итог"


def test_empty_content_is_replaced_by_reasoning_with_language_note():
    text = r._content_or_reasoning_fallback(_message("", "ход мыслей"))
    assert "ход мыслей" in text
    assert "могут быть на другом языке, чем задача" in text
    assert text.endswith("ход мыслей")


def test_empty_content_without_reasoning_stays_empty():
    assert r._content_or_reasoning_fallback(_message("", None)) == ""
    assert r._content_or_reasoning_fallback(_message(None, "")) is None


# --- форматирование ----------------------------------------------------------------------


def test_answer_is_returned_as_is_when_finished():
    assert r._format_reasoning_answer("Ответ", "stop") == "Ответ"


@pytest.mark.parametrize("answer", [None, ""])
def test_empty_answer_names_the_provider(answer):
    text = r._format_reasoning_answer(answer, "stop")
    assert r.MAIN_CLIENT_LABEL in text
    assert "пустой ответ" in text


def test_truncated_answer_has_warning():
    text = r._format_reasoning_answer("Начало", "length")
    assert text.startswith("Начало")
    assert "ОБРЕЗАН" in text


def test_intro_text_names_provider_and_is_not_investment_service():
    assert r.MAIN_CLIENT_LABEL in r.REASONING_INTRO_TEXT
    assert "не инвестиционный сервис" in r.REASONING_INTRO_TEXT


# --- клавиатуры ---------------------------------------------------------------------------


def _callbacks(keyboard) -> list[str]:
    return [b.callback_data for row in keyboard.inline_keyboard for b in row]


def test_scenario_keyboard_has_five_scenarios_new_task_and_exit():
    data = _callbacks(r._build_reasoning_scenario_keyboard())
    assert data == [
        "reasoning:scenario:1",
        "reasoning:scenario:2",
        "reasoning:scenario:3",
        "reasoning:scenario:4",
        "reasoning:scenario:5",
        "reasoning:new_task",
        "reasoning:exit",
    ]


def test_every_scenario_button_has_an_entry_in_handlers():
    for callback in _callbacks(r._build_reasoning_scenario_keyboard()):
        if ":scenario:" in callback:
            assert callback.rsplit(":", 1)[-1] in r.SCENARIO_HANDLERS


def test_expert_keyboard_has_button_per_role_and_back():
    data = _callbacks(r._build_expert_keyboard())
    assert data == [f"reasoning:expert:{key}" for key in r.EXPERT_ROLES] + [
        "reasoning:expert_back"
    ]
    assert all(len(d.encode()) <= 64 for d in data)


def test_expert_buttons_show_role_labels():
    keyboard = r._build_expert_keyboard()
    texts = [b.text for row in keyboard.inline_keyboard for b in row]
    for label, _ in r.EXPERT_ROLES.values():
        assert any(label in text for text in texts)


def test_expert_scenario_has_no_direct_handler_because_role_is_chosen_later():
    assert r.SCENARIO_HANDLERS["4"][1] is None


# --- роли и подзадачи ----------------------------------------------------------------------


def test_expert_roles_have_label_and_instruction():
    assert set(r.EXPERT_ROLES) == {"analyst", "engineer", "critic"}
    for label, instruction in r.EXPERT_ROLES.values():
        assert label
        assert instruction.startswith("Ты —")


def test_compare_all_sub_scenarios_cover_every_expert_role_once():
    experts = [extra for _l, fn, extra in r._COMPARE_ALL_SUB_SCENARIOS if extra]
    assert sorted(experts) == sorted(r.EXPERT_ROLES)
    assert len(r._COMPARE_ALL_SUB_SCENARIOS) == 3 + len(r.EXPERT_ROLES)


def test_system_prompt_requires_answer_in_task_language():
    assert "языке" in r.REASONING_SYSTEM_PROMPT


# --- запуск сценария -----------------------------------------------------------------------


def test_scenario_success_returns_answer_and_stats():
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}

    def handler(task, extra):
        return "Решение", "stop", usage

    result, error, stats = r._run_reasoning_scenario(handler, "задача")
    assert result == "Решение"
    assert error is None
    assert "finish_reason=stop" in stats
    assert "total_tokens=3" in stats


def test_scenario_passes_task_and_extra_to_handler():
    seen = []

    def handler(task, extra):
        seen.append((task, extra))
        return "a", "stop", None

    r._run_reasoning_scenario(handler, "T", "analyst")
    assert seen == [("T", "analyst")]


def test_scenario_truncated_answer_has_warning_in_result():
    result, _, _ = r._run_reasoning_scenario(lambda t, e: ("x", "length", None), "t")
    assert "ОБРЕЗАН" in result


@pytest.mark.parametrize(
    ("exc", "fragment"),
    [
        (
            AuthenticationError(
                "x", response=httpx.Response(401, request=_request()), body=None
            ),
            "авторизации",
        ),
        (
            RateLimitError("x", response=httpx.Response(429, request=_request()), body=None),
            "перегружен",
        ),
        (APIConnectionError(request=_request()), "подключиться"),
        (RuntimeError("сбой"), "непредвиденная ошибка"),
    ],
)
def test_scenario_errors_become_messages_without_result_and_stats(exc, fragment):
    result, error, stats = r._run_reasoning_scenario(_failing(exc), "t")
    assert result is None
    assert stats is None
    assert fragment in error
