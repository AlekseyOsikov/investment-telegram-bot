"""Тесты чистых правил smoke-обвязки (scripts/smoke_lib.py): разбор текстовых сценариев,
сборка апдейтов, поиск кнопок, проверка шага, HTML-скриншот и отчёт. Без бота и сети."""

import pytest
import smoke_lib as lib

SCENARIO = """
# Сценарий: Приветствие
Чат: личный
Пояснение, которое парсер игнорирует.

## Шаг: старт
Отправить: /start
Ожидать: не являюсь лицензированным
Не ожидать: гарантия

## Шаг: нажать кнопку
- Нажать: Новый профиль
- Ожидать: название
"""


def test_parse_full_scenario():
    scenario = lib.parse_scenario(SCENARIO)
    assert scenario.name == "Приветствие"
    assert scenario.chat_type == lib.CHAT_PRIVATE
    assert [step.title for step in scenario.steps] == ["старт", "нажать кнопку"]
    first, second = scenario.steps
    assert (first.action, first.argument) == (lib.ACTION_SEND, "/start")
    assert first.expect == ("не являюсь лицензированным",)
    assert first.expect_not == ("гарантия",)
    assert (second.action, second.argument) == (lib.ACTION_CLICK, "Новый профиль")


def test_parse_group_chat_and_silence():
    scenario = lib.parse_scenario(
        "# Сценарий: Группа\nЧат: групповой\n## Шаг: текст\nОтправить: привет\nТишина\n"
    )
    assert scenario.chat_type == lib.CHAT_GROUP
    assert scenario.steps[0].silent is True


@pytest.mark.parametrize(
    "text",
    [
        "## Шаг: а\nОтправить: /start\n",  # нет названия
        "# Сценарий: А\n",  # нет шагов
        "# Сценарий: А\nОтправить: /start\n",  # действие до шага
        "# Сценарий: А\n## Шаг: а\nОжидать: что-то\n",  # нет действия
        "# Сценарий: А\n## Шаг: а\nОтправить: /a\nНажать: б\n",  # два действия
        "# Сценарий: А\n## Шаг: а\nОтправить:\n",  # пустое значение
        "# Сценарий: А\nЧат: канал\n## Шаг: а\nОтправить: /a\n",  # неизвестный чат
        "# Сценарий: А\n## Шаг: а\nОтправить: /a\nТишина\nОжидать: x\n",  # тишина и ожидание
    ],
)
def test_malformed_scenarios_are_rejected(text):
    with pytest.raises(lib.ScenarioError):
        lib.parse_scenario(text)


def test_command_update_has_bot_command_entity():
    update = lib.build_message_update(7, lib.CHAT_PRIVATE, "/queue_cancel 2", now=100)
    message = update["message"]
    assert update["update_id"] == 7
    assert message["chat"] == {"id": lib.USER_ID, "type": "private"}
    assert message["entities"] == [{"type": "bot_command", "offset": 0, "length": 13}]


def test_plain_text_update_has_no_entities():
    message = lib.build_message_update(1, lib.CHAT_GROUP, "привет", now=100)["message"]
    assert "entities" not in message
    assert message["chat"] == {"id": lib.GROUP_ID, "type": "group"}


def test_callback_update_points_to_bot_message():
    update = lib.build_callback_update(
        5, lib.CHAT_PRIVATE, "pick:new", bot_message_id=42, bot_text="выбери", now=100
    )
    query = update["callback_query"]
    assert query["data"] == "pick:new"
    assert query["message"]["message_id"] == 42
    assert query["message"]["from"]["is_bot"] is True


def test_find_button_prefers_latest_and_ignores_case():
    old = lib.Reply("старое", [("➕ Новый профиль", "old")])
    new = lib.Reply("новое", [("✅ Основной", "a"), ("➕ новый ПРОФИЛЬ", "new")])
    reply, data = lib.find_button([old, new], "Новый профиль")
    assert (reply, data) == (new, "new")
    assert lib.find_button([old, new], "Нет такой") is None


def step(**changes):
    values = {"title": "т", "action": lib.ACTION_SEND, "argument": "/x"}
    return lib.Step(**{**values, **changes})


def test_check_step_passes_when_expectations_met():
    replies = [lib.Reply("Я НЕ являюсь советником")]
    assert lib.check_step(step(expect=("не являюсь",)), replies, []) == []


def test_check_step_looks_into_button_captions():
    replies = [lib.Reply("Выбери профиль:", [("➕ Новый профиль", "new")])]
    assert lib.check_step(step(expect=("новый профиль",)), replies, []) == []


def test_check_step_reports_missing_and_forbidden_text():
    replies = [lib.Reply("ответ про рынок")]
    failures = lib.check_step(step(expect=("оговорка",), expect_not=("рынок",)), replies, [])
    assert failures == ["в ответе нет «оговорка»", "в ответе есть запрещённое «рынок»"]


def test_check_step_requires_reply_unless_silent():
    assert lib.check_step(step(), [], []) == ["бот ничего не ответил"]
    assert lib.check_step(step(silent=True), [], []) == []
    assert "бот должен был молчать" in lib.check_step(step(silent=True), [lib.Reply("да")], [])[0]


def test_check_step_flags_guarantees_everywhere():
    for phrase in ("Доходность гарантированно 10%", "Точно вырастет", "рекомендуем купить"):
        failures = lib.check_step(step(), [lib.Reply(phrase)], [])
        assert any("гарантия" in item for item in failures), phrase


def test_check_step_includes_log_errors():
    failures = lib.check_step(step(), [lib.Reply("ок")], ["Traceback: KeyError"])
    assert failures == ["ошибка в логе бота: Traceback: KeyError"]


def test_action_text_marks_buttons():
    assert lib.action_text(step()) == "/x"
    assert lib.action_text(step(action=lib.ACTION_CLICK, argument="Да")) == "[кнопка] Да"


def test_html_escapes_text_and_shows_status():
    transcript = [("user", lib.Reply("/x <b>")), ("bot", lib.Reply("<script>", [("Кнопка", "d")]))]
    page = lib.render_chat_html("С", "Ш", transcript, ["сломалось <1>"])
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "Кнопка" in page
    assert "ПРОВАЛ" in page and "сломалось &lt;1&gt;" in page
    assert "OK" in lib.render_chat_html("С", "Ш", transcript, [])


def test_estimate_height_is_bounded():
    assert lib.estimate_height([], []) == 400
    huge = [("bot", lib.Reply("строка\n" * 5000))]
    assert lib.estimate_height(huge, []) == 6000


def result(passed):
    failures = [] if passed else ["в ответе нет «x»"]
    return lib.ScenarioResult(
        "Сценарий А",
        [
            lib.StepResult("шаг 1", "/start", [], [], screenshot="a/1.png"),
            lib.StepResult(
                "шаг 2", "/help", [], failures, screenshot="a/2.png", suspect="main.help_command"
            ),
        ],
    )


def test_report_summarizes_pass_and_fail():
    report = lib.render_report([result(True), result(False)], "2026-01-01")
    assert "Сценариев пройдено: 1 из 2" in report
    assert "| Сценарий А | ✅ прошёл | 2 | 0 |" in report
    assert "| Сценарий А | ❌ упал | 2 | 1 |" in report
    assert "![шаг 2](a/2.png)" in report
    assert "Что сломалось: в ответе нет «x»" in report
    assert "Где искать: main.help_command" in report


def test_report_has_no_suspect_for_passed_steps():
    assert "Где искать" not in lib.render_report([result(True)], "2026-01-01")
