"""Тесты чистых частей research/models.py (/research_models): стоимость по каталогу, сложение
стоимости, группировка подзадач по провайдерам, строки статистики, форматирование ответа,
клавиатура и перевод ошибок SDK в сообщения сценария.

Без сети, Telegram и модели; обработчик сценария подставляется простой функцией. Не покрыто
(нужны клиенты провайдеров, очередь или Telegram): call_model, call_model_kimi_k3,
call_model_deepseek_v4_pro, call_model_kimi_k2_6, call_model_deepseek_v4_flash,
call_model_compare_all, _send_scenario_result, models_command, models_receive_task,
models_scenario_callback, build_models_conversation_handler.
"""

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from research import models as m


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://example.invalid/v1")


def _status_error(cls, code: int):
    return cls("сбой", response=httpx.Response(code, request=_request()), body=None)


def _usage(prompt=1000, completion=2000, total=3000) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _ok_handler(answer="ответ", finish="stop", usage=None, elapsed=1.5, cost=0.01):
    def handler(task, extra):
        return answer, finish, usage, elapsed, cost

    return handler


def _failing_handler(exc: Exception):
    def handler(task, extra):
        raise exc

    return handler


# --- стоимость ---------------------------------------------------------------------


def test_cost_is_computed_from_catalog_prices():
    entry = m.MODEL_CATALOG[0]
    expected = (1000 * entry["price_input_per_million"] + 2000 * entry["price_output_per_million"])
    assert m._calculate_cost_usd(entry["id"], _usage()) == pytest.approx(expected / 1_000_000)


def test_cost_differs_between_models_with_different_prices():
    strong, weak = m.MODEL_CATALOG[0], m.MODEL_CATALOG[-1]
    assert m._calculate_cost_usd(strong["id"], _usage()) > m._calculate_cost_usd(
        weak["id"], _usage()
    )


@pytest.mark.parametrize("usage", [None, {}])
def test_cost_without_usage_is_none(usage):
    assert m._calculate_cost_usd(m.MODEL_CATALOG[0]["id"], usage) is None


def test_cost_of_model_outside_catalog_is_none():
    assert m._calculate_cost_usd("нет-такой-модели", _usage()) is None


def test_cost_treats_missing_or_none_token_counts_as_zero():
    entry = m.MODEL_CATALOG[0]
    only_prompt = {"prompt_tokens": 1_000_000, "completion_tokens": None}
    assert m._calculate_cost_usd(entry["id"], only_prompt) == pytest.approx(
        entry["price_input_per_million"]
    )
    assert m._calculate_cost_usd(entry["id"], {"total_tokens": 5}) == 0


def test_sum_cost_cases():
    assert m._sum_cost(None, None) is None
    assert m._sum_cost(0.5, None) == 0.5
    assert m._sum_cost(None, 0.25) == 0.25
    assert m._sum_cost(0.5, 0.25) == 0.75


def test_catalog_has_four_models_with_unique_ids_and_prices():
    assert len(m.MODEL_CATALOG) == 4
    assert len({entry["label"] for entry in m.MODEL_CATALOG}) == 4
    for entry in m.MODEL_CATALOG:
        assert entry["price_input_per_million"] > 0
        assert entry["price_output_per_million"] > 0


# --- группировка по провайдерам ----------------------------------------------------


def test_group_by_client_puts_each_provider_in_one_group_in_first_seen_order():
    groups = m._group_sub_scenarios_by_client()
    assert [[label for label, _fn in group] for group in groups] == [
        ["Kimi K3", "Kimi K2.6"],
        ["DeepSeek V4 Pro", "DeepSeek V4 Flash"],
    ]


def test_group_by_client_keeps_function_of_each_entry():
    groups = m._group_sub_scenarios_by_client()
    assert groups[0][0][1] is m.call_model_kimi_k3
    assert groups[1][1][1] is m.call_model_deepseek_v4_flash


def test_group_by_client_covers_every_sub_scenario_once():
    labels = [label for group in m._group_sub_scenarios_by_client() for label, _ in group]
    assert sorted(labels) == sorted(label for label, _c, _f in m._COMPARE_ALL_SUB_SCENARIOS)


def test_group_by_client_with_one_provider_gives_a_single_group(monkeypatch):
    shared = object()
    monkeypatch.setattr(
        m,
        "_COMPARE_ALL_SUB_SCENARIOS",
        [("a", shared, lambda t: t), ("b", shared, lambda t: t), ("c", shared, lambda t: t)],
    )
    groups = m._group_sub_scenarios_by_client()
    assert len(groups) == 1
    assert [label for label, _ in groups[0]] == ["a", "b", "c"]


def test_scenarios_1_to_4_match_catalog_order_in_keyboard_and_handlers():
    assert list(m.SCENARIO_HANDLERS) == ["1", "2", "3", "4", "5"]
    assert m.SCENARIO_HANDLERS["1"][1] is m.call_model_kimi_k3
    assert m.SCENARIO_HANDLERS["5"][1] is m.call_model_compare_all


# --- строки статистики -------------------------------------------------------------


def test_stats_line_full():
    line = m._format_stats_line("stop", _usage(10, 20, 30), 1.234, 0.0000456)
    assert line == (
        "finish_reason=stop, prompt_tokens=10, completion_tokens=20, total_tokens=30, "
        "время=1.23с, стоимость=$0.000046"
    )


def test_stats_line_without_optional_parts():
    assert m._format_stats_line(None, None, None, None) == "finish_reason=н/д"


def test_stats_line_zero_elapsed_and_zero_cost_are_shown():
    line = m._format_stats_line("stop", None, 0.0, 0.0)
    assert "время=0.00с" in line
    assert "стоимость=$0.000000" in line


def test_stats_line_with_partial_usage_marks_missing_fields():
    line = m._format_stats_line("length", {"prompt_tokens": 5}, None, None)
    assert "prompt_tokens=5" in line
    assert "completion_tokens=н/д" in line
    assert "total_tokens=н/д" in line


def test_scenario_stats_is_marked_stats_line():
    assert m._format_scenario_stats("stop", None, 2.0, None) == (
        "📈 " + m._format_stats_line("stop", None, 2.0, None)
    )


# --- ответ -------------------------------------------------------------------------


def test_model_answer_is_returned_as_is_for_normal_finish():
    assert m._format_model_answer("Ответ.", "stop") == "Ответ."


@pytest.mark.parametrize("answer", [None, ""])
def test_empty_model_answer_gets_placeholder(answer):
    assert m._format_model_answer(answer, "stop") == "Модель вернула пустой ответ."


def test_truncated_answer_gets_a_warning_after_the_text():
    text = m._format_model_answer("Начало", "length")
    assert text.startswith("Начало")
    assert "ОБРЕЗАН" in text


def test_empty_truncated_answer_has_placeholder_and_warning():
    text = m._format_model_answer(None, "length")
    assert "пустой ответ" in text
    assert "ОБРЕЗАН" in text


# --- клавиатура --------------------------------------------------------------------


def test_keyboard_has_five_scenarios_new_task_and_exit_with_models_prefix():
    keyboard = m._build_models_scenario_keyboard()
    data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert data == [
        "models:scenario:1",
        "models:scenario:2",
        "models:scenario:3",
        "models:scenario:4",
        "models:scenario:5",
        "models:new_task",
        "models:exit",
    ]
    assert all(len(d.encode()) <= 64 for d in data)


def test_every_scenario_button_has_a_handler():
    keyboard = m._build_models_scenario_keyboard()
    for row in keyboard.inline_keyboard:
        for button in row:
            if button.callback_data.startswith("models:scenario:"):
                assert button.callback_data.rsplit(":", 1)[-1] in m.SCENARIO_HANDLERS


# --- запуск сценария ---------------------------------------------------------------


def test_scenario_success_returns_formatted_answer_and_stats_without_error():
    result, error, stats = m._run_models_scenario(
        _ok_handler("Текст", "stop", _usage(1, 2, 3), 0.5, 0.001), "задача"
    )
    assert result == "Текст"
    assert error is None
    assert stats == m._format_scenario_stats("stop", _usage(1, 2, 3), 0.5, 0.001)


def test_scenario_passes_task_and_extra_to_the_handler():
    seen = []

    def handler(task, extra):
        seen.append((task, extra))
        return "a", "stop", None, 0.1, None

    m._run_models_scenario(handler, "T", "E")
    assert seen == [("T", "E")]


def test_scenario_truncated_answer_has_warning():
    result, _, _ = m._run_models_scenario(_ok_handler("x", "length"), "t")
    assert "ОБРЕЗАН" in result


@pytest.mark.parametrize(
    ("exc", "fragment"),
    [
        (_status_error(AuthenticationError, 401), "авторизации"),
        (_status_error(RateLimitError, 429), "перегружен"),
        (APITimeoutError(request=_request()), "не ответила вовремя"),
        (TimeoutError(), "не ответила вовремя"),
        (APIConnectionError(request=_request()), "подключиться"),
        (_status_error(APIStatusError, 404), "модель недоступна"),
        (RuntimeError("что-то"), "непредвиденная ошибка"),
    ],
)
def test_scenario_errors_become_neutral_russian_messages(exc, fragment):
    result, error, stats = m._run_models_scenario(_failing_handler(exc), "t")
    assert result is None
    assert stats is None
    assert fragment in error


def test_error_messages_never_name_a_specific_provider():
    excs = [
        _status_error(AuthenticationError, 401),
        _status_error(RateLimitError, 429),
        APITimeoutError(request=_request()),
        APIConnectionError(request=_request()),
        _status_error(APIStatusError, 500),
        RuntimeError("x"),
    ]
    for exc in excs:
        _, error, _ = m._run_models_scenario(_failing_handler(exc), "t")
        assert "DeepSeek" not in error
        assert "Kimi" not in error
