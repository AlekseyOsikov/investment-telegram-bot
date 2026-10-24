"""Тесты чистых частей research/constraints.py (/research_constraints): снятие обёртки
```json, проверка и форматирование ответа каждого сценария (в том числе отказы при невалидном
JSON), клавиатура, запуск сценария с подставленным обработчиком, константы формата.

Без сети, Telegram и модели. Не покрыто (нужны модель, очередь или Telegram):
call_constraint_no_restrictions, call_constraint_json_schema, call_constraint_max_tokens,
call_constraint_stop_sequence, _send_scenario_result, constraints_command,
constraints_receive_question, constraints_scenario_callback, constraints_max_tokens_input,
constraints_stop_words_input, build_constraints_conversation_handler.
"""

import json

import pytest

from research import constraints as c


def _company(**overrides) -> dict:
    item = {
        "ticker": "DEMO",
        "company_name": "Демо Компания",
        "sector": "Тест",
        "summary": "Краткое описание.",
    }
    item.update(overrides)
    return item


def _json_answer(companies) -> str:
    return json.dumps({c.CONSTRAINTS_JSON_LIST_KEY: companies}, ensure_ascii=False)


def _is_refusal(text: str) -> bool:
    return text.startswith("⚠️ Введённый запрос не может быть обработан")


# --- _strip_code_fence --------------------------------------------------------------------


def test_fence_with_language_is_removed():
    assert c._strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_fence_without_language_is_removed():
    assert c._strip_code_fence("```\n[1, 2]\n```") == "[1, 2]"


def test_plain_text_is_only_stripped():
    assert c._strip_code_fence('  {"a": 1}\n') == '{"a": 1}'


def test_fence_without_closing_marker_still_loses_opening_line():
    assert c._strip_code_fence('```json\n{"a": 1}') == '{"a": 1}'


def test_single_line_fence_loses_only_the_marker():
    assert c._strip_code_fence("```json") == "json"


def test_empty_text_stays_empty():
    assert c._strip_code_fence("") == ""
    assert c._strip_code_fence("   \n ") == ""


# --- сценарий 2: JSON ----------------------------------------------------------------------


def test_valid_json_is_pretty_printed_in_a_json_block_keeping_cyrillic():
    text = c._format_constraints_answer("2", _json_answer([_company()]), "stop")
    assert text.startswith("```json\n")
    assert text.endswith("\n```")
    assert "Демо Компания" in text
    assert json.loads(text.removeprefix("```json\n").removesuffix("\n```")) == {
        c.CONSTRAINTS_JSON_LIST_KEY: [_company()]
    }


def test_json_wrapped_in_code_fence_by_the_model_is_accepted():
    answer = "```json\n" + _json_answer([_company()]) + "\n```"
    assert c._format_constraints_answer("2", answer, "stop").startswith("```json\n")


def test_several_companies_are_all_kept():
    answer = _json_answer([_company(ticker="AAA"), _company(ticker="BBB")])
    text = c._format_constraints_answer("2", answer, "stop")
    assert "AAA" in text and "BBB" in text


def test_extra_fields_are_allowed_in_company_objects():
    answer = _json_answer([_company(extra="x")])
    assert c._format_constraints_answer("2", answer, "stop").startswith("```json\n")


@pytest.mark.parametrize("answer", [None, ""])
def test_empty_answer_in_json_scenario_is_refused_naming_provider(answer):
    text = c._format_constraints_answer("2", answer, "stop")
    assert _is_refusal(text)
    assert c.MAIN_CLIENT_LABEL in text
    assert "пустой ответ" in text


@pytest.mark.parametrize(
    "answer",
    [
        "это не JSON",
        "{обрезанный",
        "[1, 2, 3]",  # корень — не объект
        '"строка"',
        "5",
        '{"other": []}',  # нет ключа списка
        '{"companies": "x"}',  # не список
        '{"companies": []}',  # пустой список
        '{"companies": ["строка"]}',  # элемент — не объект
        json.dumps({"companies": [_company(), 5]}),
    ],
)
def test_json_with_invalid_structure_is_refused(answer):
    text = c._format_constraints_answer("2", answer, "stop")
    assert _is_refusal(text)
    assert "невалидный JSON" in text


def test_missing_fields_are_named_in_refusal():
    company = _company()
    del company["sector"]
    del company["summary"]
    text = c._format_constraints_answer("2", _json_answer([company]), "stop")
    assert _is_refusal(text)
    assert "sector, summary" in text
    assert "ticker" not in text.split("(")[1]


def test_missing_fields_are_checked_for_every_company():
    broken = _company()
    del broken["ticker"]
    text = c._format_constraints_answer("2", _json_answer([_company(), broken]), "stop")
    assert _is_refusal(text)
    assert "ticker" in text


def test_json_scenario_ignores_finish_reason_length():
    text = c._format_constraints_answer("2", _json_answer([_company()]), "length")
    assert "ОБРЕЗАН" not in text


# --- остальные сценарии --------------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", ["1", "3", "4"])
def test_plain_scenarios_return_answer_as_is(scenario_id):
    assert c._format_constraints_answer(scenario_id, "Ответ", "stop") == "Ответ"


@pytest.mark.parametrize("scenario_id", ["1", "3", "4"])
def test_plain_scenarios_mark_empty_answer(scenario_id):
    for answer in (None, ""):
        text = c._format_constraints_answer(scenario_id, answer, "stop")
        assert c.MAIN_CLIENT_LABEL in text
        assert "пустой ответ" in text


@pytest.mark.parametrize("scenario_id", ["1", "3", "4"])
def test_plain_scenarios_warn_when_truncated(scenario_id):
    text = c._format_constraints_answer(scenario_id, "Начало", "length")
    assert text.startswith("Начало")
    assert "ОБРЕЗАН" in text


def test_stop_finish_reason_has_no_truncation_warning():
    assert "ОБРЕЗАН" not in c._format_constraints_answer("3", "x", "stop")


# --- клавиатура и константы ------------------------------------------------------------------


def test_keyboard_has_four_scenarios_new_question_and_exit():
    keyboard = c._build_constraints_keyboard()
    data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert data == [
        "constraints:scenario:1",
        "constraints:scenario:2",
        "constraints:scenario:3",
        "constraints:scenario:4",
        "constraints:new_question",
        "constraints:exit",
    ]


def test_every_scenario_button_has_a_handler():
    assert sorted(c.SCENARIO_HANDLERS) == ["1", "2", "3", "4"]


def test_intro_text_has_not_a_financial_advisor_reminder_and_provider():
    assert "не является лицензированным финансовым советником" in c.CONSTRAINTS_INTRO_TEXT
    assert "не инвестиционная рекомендация" in c.CONSTRAINTS_INTRO_TEXT
    assert c.MAIN_CLIENT_LABEL in c.CONSTRAINTS_INTRO_TEXT


def test_json_instruction_names_list_key_and_every_required_field():
    assert f'"{c.CONSTRAINTS_JSON_LIST_KEY}"' in c.CONSTRAINTS_JSON_INSTRUCTION
    for field in c.CONSTRAINTS_JSON_FIELDS:
        assert f'"{field}"' in c.CONSTRAINTS_JSON_INSTRUCTION


def test_json_response_format_is_object_mode_not_schema():
    assert c.CONSTRAINTS_JSON_RESPONSE_FORMAT == {"type": "json_object"}


def test_stop_word_limit_is_four():
    assert c.CONSTRAINTS_MAX_STOP_WORDS == 4


# --- запуск сценария -------------------------------------------------------------------------


def _patch_handler(monkeypatch, scenario_id, fn):
    monkeypatch.setitem(c.SCENARIO_HANDLERS, scenario_id, ("тест", fn))


def test_run_scenario_formats_json_answer_and_builds_stats(monkeypatch):
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    _patch_handler(
        monkeypatch, "2", lambda question, extra: (_json_answer([_company()]), "stop", usage)
    )
    result, error, stats = c._run_constraints_scenario("2", "вопрос")
    assert result.startswith("```json\n")
    assert error is None
    assert "total_tokens=3" in stats


def test_run_scenario_passes_question_and_extra(monkeypatch):
    seen = []

    def handler(question, extra):
        seen.append((question, extra))
        return "ответ", "stop", None

    _patch_handler(monkeypatch, "3", handler)
    result, _, _ = c._run_constraints_scenario("3", "вопрос", 50)
    assert seen == [("вопрос", 50)]
    assert result == "ответ"


def test_run_scenario_uses_formatting_of_its_own_scenario(monkeypatch):
    _patch_handler(monkeypatch, "1", lambda q, e: ("x", "length", None))
    result, _, _ = c._run_constraints_scenario("1", "q")
    assert "ОБРЕЗАН" in result


def test_run_scenario_error_becomes_message(monkeypatch):
    def handler(question, extra):
        raise RuntimeError("сбой")

    _patch_handler(monkeypatch, "4", handler)
    result, error, stats = c._run_constraints_scenario("4", "q", ["x"])
    assert result is None
    assert stats is None
    assert "непредвиденная ошибка" in error


def test_run_scenario_unknown_id_is_key_error():
    with pytest.raises(KeyError):
        c._run_constraints_scenario("9", "q")
