"""Тесты чистых функций общего каркаса research-режимов (research/_shared.py): статистика токенов,
подстановка рассуждений вместо пустого ответа, перевод ошибок OpenAI SDK в русские сообщения,
запуск сценария с форматированием.

Без сети, Telegram и модели; обработчик /cancel (build_cancel_handler) проверяется на заглушках
update/context.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from telegram.ext import ConversationHandler

from research import _shared as sh


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://example.invalid/v1")


def _status(cls, code: int):
    return cls("сбой", response=httpx.Response(code, request=_request()), body=None)


def _usage(prompt=10, completion=20, total=30) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


# --- extract_usage ----------------------------------------------------------------------


def test_extract_usage_reads_three_counters():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    )
    assert sh.extract_usage(response) == _usage(1, 2, 3)


def test_extract_usage_without_usage_is_none():
    assert sh.extract_usage(SimpleNamespace(usage=None)) is None
    assert sh.extract_usage(SimpleNamespace()) is None


def test_extract_usage_with_missing_counters_gives_none_values():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5))
    assert sh.extract_usage(response) == {
        "prompt_tokens": 5,
        "completion_tokens": None,
        "total_tokens": None,
    }


# --- content_or_reasoning_fallback --------------------------------------------------------


def test_visible_content_wins_over_reasoning():
    message = SimpleNamespace(content="Ответ", reasoning_content="мысли")
    assert sh.content_or_reasoning_fallback(message) == "Ответ"


def test_empty_content_is_replaced_by_marked_reasoning():
    message = SimpleNamespace(content="", reasoning_content="ход мыслей")
    text = sh.content_or_reasoning_fallback(message)
    assert text.startswith("[Модель ещё не начала видимый ответ, вот её рассуждения]")
    assert text.endswith("ход мыслей")


def test_reasoning_note_is_inserted_into_the_marker():
    message = SimpleNamespace(content=None, reasoning_content="мысли")
    text = sh.content_or_reasoning_fallback(message, reasoning_note=" (на другом языке)")
    assert text.startswith("[Модель ещё не начала видимый ответ, вот её рассуждения (на другом")


def test_empty_content_without_reasoning_is_returned_as_is():
    assert sh.content_or_reasoning_fallback(SimpleNamespace(content="")) == ""
    assert sh.content_or_reasoning_fallback(SimpleNamespace(content=None)) is None
    message = SimpleNamespace(content=None, reasoning_content="")
    assert sh.content_or_reasoning_fallback(message) is None


# --- sum_usage ----------------------------------------------------------------------------


def test_sum_usage_of_nothing_is_none():
    assert sh.sum_usage(None, None) is None


def test_sum_usage_with_one_side_returns_that_side():
    assert sh.sum_usage(_usage(1, 2, 3), None) == _usage(1, 2, 3)
    assert sh.sum_usage(None, _usage(4, 5, 9)) == _usage(4, 5, 9)


def test_sum_usage_adds_each_counter():
    assert sh.sum_usage(_usage(1, 2, 3), _usage(10, 20, 30)) == _usage(11, 22, 33)


def test_sum_usage_skips_counters_missing_on_both_sides_and_treats_one_sided_as_zero():
    assert sh.sum_usage({"prompt_tokens": 1}, {"prompt_tokens": 2}) == {"prompt_tokens": 3}
    assert sh.sum_usage({"prompt_tokens": 1}, {"total_tokens": 5}) == {
        "prompt_tokens": 1,
        "total_tokens": 5,
    }


def test_sum_usage_of_empty_dicts_is_none():
    assert sh.sum_usage({}, {}) is None


def test_sum_usage_does_not_modify_arguments():
    first, second = _usage(1, 2, 3), _usage(1, 2, 3)
    sh.sum_usage(first, second)
    assert first == _usage(1, 2, 3)
    assert second == _usage(1, 2, 3)


# --- format_scenario_stats ----------------------------------------------------------------


def test_stats_with_usage():
    assert sh.format_scenario_stats("stop", _usage(1, 2, 3)) == (
        "📈 finish_reason=stop, prompt_tokens=1, completion_tokens=2, total_tokens=3"
    )


def test_stats_without_finish_reason_or_usage():
    assert sh.format_scenario_stats(None, None) == "📈 finish_reason=н/д"
    assert sh.format_scenario_stats("", {}) == "📈 finish_reason=н/д"


def test_stats_with_partial_usage_marks_missing_counters():
    text = sh.format_scenario_stats("length", {"prompt_tokens": 7})
    assert "prompt_tokens=7" in text
    assert "completion_tokens=н/д" in text
    assert "total_tokens=н/д" in text


# --- api_error_to_message -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "fragment"),
    [
        (_status(AuthenticationError, 401), "Ошибка авторизации"),
        (_status(RateLimitError, 429), "превышен лимит запросов"),
        (APITimeoutError(request=_request()), "не ответил вовремя"),
        (TimeoutError(), "не ответил вовремя"),
        (APIConnectionError(request=_request()), "Не получилось подключиться"),
        (_status(APIStatusError, 500), "вернул ошибку"),
    ],
)
def test_sdk_errors_become_russian_messages_with_provider_label(exc, fragment):
    message = sh.api_error_to_message(exc, "Тестпровайдер", "TEST_API_KEY")
    assert fragment in message
    assert "Тестпровайдер" in message


def test_non_sdk_error_is_not_translated():
    assert sh.api_error_to_message(ValueError("x"), "П", "K") is None
    assert sh.api_error_to_message(RuntimeError("x"), "П", "K") is None


def test_auth_error_does_not_leak_the_key_variable_to_the_user_text():
    message = sh.api_error_to_message(_status(AuthenticationError, 401), "П", "SECRET_ENV_NAME")
    assert "SECRET_ENV_NAME" not in message
    assert "API-ключ" in message


def test_timeout_is_checked_before_connection_error():
    # APITimeoutError — подкласс APIConnectionError: должно сработать сообщение про тайм-аут.
    message = sh.api_error_to_message(APITimeoutError(request=_request()), "П", "K")
    assert "не ответил вовремя" in message
    assert "подключиться" not in message


# --- run_scenario -------------------------------------------------------------------------


def _format(answer, finish_reason):
    return f"<{answer}|{finish_reason}>"


def test_run_scenario_success_returns_formatted_answer_and_stats_without_error():
    def handler(a, b):
        return "ответ", "stop", _usage(1, 2, 3)

    formatted, error, stats = sh.run_scenario(handler, 1, 2, format_answer=_format)
    assert formatted == "<ответ|stop>"
    assert error is None
    assert stats == sh.format_scenario_stats("stop", _usage(1, 2, 3))


def test_run_scenario_passes_positional_arguments():
    seen = []

    def handler(*args):
        seen.append(args)
        return "a", "stop", None

    sh.run_scenario(handler, "задача", {"x": 1}, format_answer=_format)
    assert seen == [("задача", {"x": 1})]


def test_run_scenario_sdk_error_uses_given_label():
    def handler():
        raise _status(RateLimitError, 429)

    formatted, error, stats = sh.run_scenario(
        handler, format_answer=_format, label="Другой", api_key_env_var="K"
    )
    assert formatted is None
    assert stats is None
    assert "Другой" in error
    assert "лимит" in error


def test_run_scenario_default_label_is_the_main_provider():
    def handler():
        raise APITimeoutError(request=_request())

    _, error, _ = sh.run_scenario(handler, format_answer=_format)
    assert sh.MAIN_CLIENT_LABEL in error


def test_run_scenario_unexpected_error_gives_generic_message_and_logs(caplog):
    def handler():
        raise RuntimeError("внутренний сбой")

    formatted, error, stats = sh.run_scenario(handler, format_answer=_format)
    assert (formatted, stats) == (None, None)
    assert error == "❌ Произошла непредвиденная ошибка. Попробуй ещё раз чуть позже."
    assert "внутренний сбой" not in error
    assert "внутренний сбой" in caplog.text


# --- build_cancel_handler -----------------------------------------------------------------


def _cancel_call(user_data: dict, key: str):
    replies = []

    async def reply_text(text):
        replies.append(text)

    update = SimpleNamespace(message=SimpleNamespace(reply_text=reply_text))
    context = SimpleNamespace(user_data=user_data)
    result = asyncio.run(sh.build_cancel_handler(key)(update, context))
    return result, replies


def test_cancel_clears_only_own_key_replies_and_ends_conversation():
    user_data = {"reasoning_task": "x", "other": 1}
    result, replies = _cancel_call(user_data, "reasoning_task")
    assert result == ConversationHandler.END
    assert user_data == {"other": 1}
    assert replies == ["Исследование прервано. Возвращаюсь в обычный режим."]


def test_cancel_without_stored_key_does_not_fail():
    result, replies = _cancel_call({}, "missing")
    assert result == ConversationHandler.END
    assert len(replies) == 1
