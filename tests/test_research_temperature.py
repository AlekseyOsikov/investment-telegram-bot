"""Тесты чистых частей research/temperature.py (/research_temperature): фиксированные значения
temperature у сценариев 1-4, форматирование ответа, клавиатура, запуск сценария с подставленным
обработчиком, состав подзадач сравнения.

Без сети, Telegram и модели: общий вызов `call_temperature` подменяется записью аргументов.
Не покрыто (нужны модель, очередь или Telegram): call_temperature (запрос к API),
call_temperature_compare_all, _send_scenario_result, temperature_command,
temperature_receive_task, temperature_scenario_callback, build_temperature_conversation_handler.
"""

import httpx
import pytest
from openai import RateLimitError

from research import temperature as t


def _callbacks(keyboard) -> list[str]:
    return [b.callback_data for row in keyboard.inline_keyboard for b in row]


# --- значения temperature ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (t.call_temperature_0, 0),
        (t.call_temperature_0_7, 0.7),
        (t.call_temperature_1_2, 1.2),
        (t.call_temperature_2, 2),
    ],
)
def test_fixed_scenarios_pass_their_temperature_and_the_task(monkeypatch, fn, expected):
    calls = []

    def recorder(task, temperature):
        calls.append((task, temperature))
        return "ответ", "stop", None

    monkeypatch.setattr(t, "call_temperature", recorder)
    assert fn("задача", "лишнее") == ("ответ", "stop", None)
    assert calls == [("задача", expected)]


def test_scenarios_1_to_4_are_ordered_by_growing_temperature():
    labels = [t.SCENARIO_HANDLERS[key][0] for key in "1234"]
    assert labels == [
        "temperature = 0",
        "temperature = 0.7",
        "temperature = 1.2",
        "temperature = 2",
    ]
    assert t.SCENARIO_HANDLERS["5"][1] is t.call_temperature_compare_all


def test_compare_all_covers_the_same_four_calls_as_scenarios_1_to_4():
    sub = [(label, fn) for label, fn, _ in t._COMPARE_ALL_SUB_SCENARIOS]
    single = [t.SCENARIO_HANDLERS[key] for key in "1234"]
    assert sub == single


def test_compare_instruction_names_all_values_and_three_criteria():
    for value in ("0", "0.7", "1.2", "2"):
        assert value in t.COMPARE_ALL_INSTRUCTION
    for criterion in ("Точность", "Креативность", "Разнообразие"):
        assert criterion in t.COMPARE_ALL_INSTRUCTION


def test_intro_text_names_provider_and_is_not_investment_service():
    assert t.MAIN_CLIENT_LABEL in t.TEMPERATURE_INTRO_TEXT
    assert "не инвестиционный сервис" in t.TEMPERATURE_INTRO_TEXT


def test_system_prompt_is_neutral_and_requires_task_language():
    assert "языке" in t.TEMPERATURE_SYSTEM_PROMPT
    assert "инвест" not in t.TEMPERATURE_SYSTEM_PROMPT.lower()


# --- форматирование ---------------------------------------------------------------------


def test_answer_is_returned_as_is_when_finished():
    assert t._format_temperature_answer("Ответ", "stop") == "Ответ"


@pytest.mark.parametrize("answer", [None, ""])
def test_empty_answer_names_the_provider(answer):
    text = t._format_temperature_answer(answer, "stop")
    assert t.MAIN_CLIENT_LABEL in text
    assert "пустой ответ" in text


def test_truncated_answer_has_warning_after_text():
    text = t._format_temperature_answer("Начало", "length")
    assert text.startswith("Начало")
    assert "ОБРЕЗАН" in text


# --- клавиатура -------------------------------------------------------------------------


def test_keyboard_has_five_scenarios_new_task_and_exit():
    assert _callbacks(t._build_temperature_scenario_keyboard()) == [
        "temperature:scenario:1",
        "temperature:scenario:2",
        "temperature:scenario:3",
        "temperature:scenario:4",
        "temperature:scenario:5",
        "temperature:new_task",
        "temperature:exit",
    ]


def test_every_scenario_button_has_a_handler():
    for callback in _callbacks(t._build_temperature_scenario_keyboard()):
        if ":scenario:" in callback:
            assert callback.rsplit(":", 1)[-1] in t.SCENARIO_HANDLERS


# --- запуск сценария --------------------------------------------------------------------


def test_scenario_success_returns_answer_and_stats():
    usage = {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
    result, error, stats = t._run_temperature_scenario(
        lambda task, extra: ("Решение", "stop", usage), "задача"
    )
    assert result == "Решение"
    assert error is None
    assert "total_tokens=9" in stats


def test_scenario_passes_task_and_extra_to_handler():
    seen = []

    def handler(task, extra):
        seen.append((task, extra))
        return "a", "stop", None

    t._run_temperature_scenario(handler, "T", "E")
    assert seen == [("T", "E")]


def test_scenario_truncated_answer_has_warning_in_result():
    result, _, _ = t._run_temperature_scenario(lambda task, extra: ("x", "length", None), "t")
    assert "ОБРЕЗАН" in result


def test_scenario_api_error_becomes_message_without_result_and_stats():
    response = httpx.Response(429, request=httpx.Request("POST", "http://example.invalid"))

    def handler(task, extra):
        raise RateLimitError("лимит", response=response, body=None)

    result, error, stats = t._run_temperature_scenario(handler, "t")
    assert result is None
    assert stats is None
    assert "перегружен" in error


def test_scenario_unexpected_error_becomes_generic_message():
    def handler(task, extra):
        raise RuntimeError("сбой")

    result, error, stats = t._run_temperature_scenario(handler, "t")
    assert (result, stats) == (None, None)
    assert "непредвиденная ошибка" in error
