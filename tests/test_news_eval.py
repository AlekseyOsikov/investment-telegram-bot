"""Тесты правил оценки разбора новостей (research/news_eval.py) без сети; модель — фейк."""

import json

import pytest

from news_digest import rules
from research import news_eval as ev

GOOD = {
    "event_type": "dividends",
    "importance": "medium",
    "summary": "Совет директоров рекомендовал выплатить дивиденды.",
    "mechanism": "Дивидендная отсечка уменьшает стоимость акции на размер выплаты.",
}
LINK = "https://example.com/n/1"


def reply(**changes):
    return json.dumps({**GOOD, **changes}, ensure_ascii=False)


def dataset(**extra):
    return ev.parse_dataset(
        {
            "name": "тест",
            "items": [
                {
                    "title": "Альфатрейд объявила дивиденды",
                    "link": LINK,
                    "tickers": ["alfa"],
                    **extra,
                }
            ],
        }
    )


def test_parse_dataset_normalizes_and_validates():
    parsed = dataset(expected_type="dividends")
    item = parsed.items[0]
    assert parsed.name == "тест" and item.tickers == ("ALFA",) and item.expected_type == "dividends"


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"items": "нет"},
        {"items": []},
        {"items": [{"excerpt": "без заголовка"}]},
        {"items": [{"title": "Н", "expected_type": "придумано"}]},
        {"items": [{"title": "Н", "tickers": "ALFA"}]},
    ],
)
def test_parse_dataset_rejects_bad_format(data):
    with pytest.raises(ev.DatasetError):
        ev.parse_dataset(data)


def test_evaluate_accepts_correct_answer_and_matches_type():
    item = dataset(expected_type="dividends").items[0]
    result = ev.evaluate_item(item, 1, lambda messages: reply(), "ОСНОВА")
    assert result.ok and result.attempts == 1 and result.type_matches is True


def test_evaluate_marks_type_mismatch():
    item = dataset(expected_type="reporting").items[0]
    assert ev.evaluate_item(item, 1, lambda m: reply(), None).type_matches is False


def test_unlabeled_item_has_no_type_verdict():
    assert ev.evaluate_item(dataset().items[0], 1, lambda m: reply(), None).type_matches is None


def test_violation_then_retry_uses_same_checks_as_bot():
    answers = iter([reply(summary="Акция вырастет."), reply()])
    result = ev.evaluate_item(dataset().items[0], 1, lambda m: next(answers), None)
    assert result.ok and result.attempts == 2


def test_persistent_violation_and_missing_link_are_reported():
    result = ev.evaluate_item(
        dataset().items[0], 1, lambda m: reply(mechanism="Рекомендуем купить."), None
    )
    assert not result.ok and result.attempts == rules.MAX_ATTEMPTS
    assert any("запрещённые" in p for p in result.problems)
    no_link = ev.parse_dataset({"items": [{"title": "Без ссылки"}]}).items[0]
    bad = ev.evaluate_item(no_link, 1, lambda m: reply(), None)
    assert not bad.ok and any("ссылк" in p for p in bad.problems)


def test_model_error_is_recorded_not_raised():
    def broken(messages):
        raise ConnectionError("нет связи")

    result = ev.evaluate_item(dataset().items[0], 1, broken, None)
    assert not result.ok and "ConnectionError" in result.error and result.attempts == 1


def test_summary_and_report():
    item_ok = dataset(expected_type="dividends").items[0]
    item_bad = dataset().items[0]
    results = [
        ev.evaluate_item(item_ok, 1, lambda m: reply(), None),
        ev.evaluate_item(item_bad, 2, lambda m: reply(summary="Это хорошая новость."), None),
        ev.evaluate_item(item_bad, 3, lambda m: (_ for _ in ()).throw(TimeoutError("t")), None),
    ]
    summary = ev.summarize(results)
    assert (summary.total, summary.accepted, summary.first_try) == (3, 1, 1)
    assert (summary.fallback, summary.errors) == (1, 1)
    assert summary.labeled == 1 and summary.type_share == 1.0
    assert summary.violations == {"запрещённые формулировки": 1}
    report = ev.render_report(dataset(), results, summary, "тест-модель")
    assert "тест-модель" in report and "100%" in report and "запрещённые формулировки: 1" in report
    assert "ошибка модели" in report and "ничего не меняет" in report


def test_empty_results_summary():
    summary = ev.summarize([])
    assert summary.total == 0 and summary.type_share is None


# --- Дополнение: группы нарушений, пустой цикл, отчёт без разметки, формат новости -------------


def test_problem_key_groups_real_validation_problems():
    long_summary = "я" * (rules.MAX_SUMMARY_CHARS + 1)
    cases = {
        "в поле summary запрещённые формулировки": reply(summary="Это хорошая новость."),
        "ответ не JSON": "это не JSON",
        "тип события не из перечня": reply(event_type="придумано"),
        "важность не из перечня": reply(importance="огромная"),
        "слишком длинное поле": reply(summary=long_summary),
        "пустое поле": reply(mechanism=""),
    }
    expected_keys = {
        "в поле summary запрещённые формулировки": "запрещённые формулировки",
        "ответ не JSON": "ответ не JSON",
        "тип события не из перечня": "тип события не из перечня",
        "важность не из перечня": "важность не из перечня",
        "слишком длинное поле": "слишком длинное поле",
        "пустое поле": "пустое поле",
    }
    for name, raw in cases.items():
        validation = rules.validate_analysis(raw, LINK)
        assert validation.analysis is None, name
        assert {ev._problem_key(p) for p in validation.problems} == {expected_keys[name]}, name


def test_problem_key_for_missing_link_and_unknown_problem():
    validation = rules.validate_analysis(reply(), "")
    assert [ev._problem_key(p) for p in validation.problems] == ["нет ссылки"]
    assert ev._problem_key("что-то новое") == "что-то новое"


def test_evaluate_item_without_attempts_returns_empty_result(monkeypatch):
    monkeypatch.setattr(rules, "MAX_ATTEMPTS", 0)
    calls = []
    result = ev.evaluate_item(dataset().items[0], 1, lambda m: calls.append(m) or reply(), None)
    assert calls == []
    assert result.attempts == 0
    assert not result.ok
    assert result.error == "" and result.problems == []


def test_evaluate_item_first_try_and_retry_attempt_counts():
    item = dataset().items[0]
    assert ev.evaluate_item(item, 1, lambda m: reply(), None).attempts == 1
    answers = iter([reply(summary="Это хорошая новость."), reply()])
    retried = ev.evaluate_item(item, 2, lambda m: next(answers), None)
    assert retried.ok
    assert retried.attempts == 2
    assert retried.problems == []


def test_second_attempt_receives_problems_of_the_first_in_messages():
    item = dataset().items[0]
    seen = []
    answers = iter([reply(event_type="придумано"), reply()])

    def complete(messages):
        seen.append(messages)
        return next(answers)

    ev.evaluate_item(item, 1, complete, None)
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert "event_type" in json.dumps(seen[1], ensure_ascii=False)


def test_to_news_has_bot_record_shape_and_ticker_binding():
    item = ev.EvalItem(
        title="Заголовок",
        excerpt="Выдержка",
        source="лента",
        link=LINK,
        published_at="2026-10-01T10:00:00+00:00",
        tickers=("ALFA", "BETA"),
    )
    news = ev.to_news(item, 3)
    assert news["id"] == "eval-3"
    assert (news["source"], news["link"], news["title"], news["excerpt"]) == (
        "лента",
        LINK,
        "Заголовок",
        "Выдержка",
    )
    assert news["published_at"] == "2026-10-01T10:00:00+00:00"
    assert news["tickers"] == [
        {"ticker": "ALFA", "matched": "ALFA", "kind": "ticker"},
        {"ticker": "BETA", "matched": "BETA", "kind": "ticker"},
    ]
    assert ev.to_news(ev.EvalItem(title="Т"), 1)["tickers"] == []


def test_parse_dataset_defaults_and_trimming():
    parsed = ev.parse_dataset(
        {"items": [{"title": "  Заголовок  ", "tickers": [" alfa ", "Beta"]}, {"title": "Второй"}]},
        default_name="имя файла",
    )
    first, second = parsed.items
    assert parsed.name == "имя файла"
    assert first.title == "Заголовок"
    assert first.tickers == ("ALFA", "BETA")
    assert (first.source, first.excerpt, first.link, first.published_at) == ("набор", "", "", "")
    assert first.expected_type is None
    assert second.tickers == ()


def test_parse_dataset_errors_name_the_item_number():
    with pytest.raises(ev.DatasetError, match="№2"):
        ev.parse_dataset({"items": [{"title": "Ок"}, {"title": "  "}]})
    with pytest.raises(ev.DatasetError, match="№1.*tickers"):
        ev.parse_dataset({"items": [{"title": "Ок", "tickers": [1]}]})
    with pytest.raises(ev.DatasetError, match="expected_type"):
        ev.parse_dataset({"items": [{"title": "Ок", "expected_type": "придумано"}]})


def test_item_result_type_matches_is_none_without_labels_or_analysis():
    item = ev.EvalItem(title="Т", expected_type="dividends")
    assert ev.ItemResult(item).type_matches is None
    assert ev.ItemResult(ev.EvalItem(title="Т"), analysis=dict(GOOD)).type_matches is None
    assert ev.ItemResult(item, analysis=dict(GOOD)).type_matches is True
    other = ev.ItemResult(item, analysis={**GOOD, "event_type": "other"})
    assert other.type_matches is False


def test_report_without_labels_says_share_is_not_counted_and_marks_mismatch_and_fallback():
    labeled = dataset(expected_type="reporting").items[0]
    unlabeled = dataset().items[0]
    mismatch = ev.evaluate_item(labeled, 1, lambda m: reply(), None)
    plain = ev.evaluate_item(unlabeled, 2, lambda m: reply(), None)
    rejected = ev.evaluate_item(unlabeled, 3, lambda m: reply(summary="Это хорошая новость."), None)

    only_plain = ev.summarize([plain])
    text = ev.render_report(dataset(), [plain], only_plain, "м")
    assert "Разметки типа нет — доля совпадения не считается." in text
    assert "Нарушения в отвергнутых ответах" not in text

    results = [mismatch, plain, rejected]
    text = ev.render_report(dataset(), results, ev.summarize(results), "м")
    assert "тип НЕ совпал (ожидался reporting)" in text
    assert "принято (попыток: 1), dividends/medium\n" in text
    assert "отвергнуто:" in text
    assert "Совпадение типа события с разметкой: 0 из 1 (0%)." in text
    assert text.endswith("остаются решением оператора.")


def test_summary_counts_first_try_separately_from_retries_and_ignores_problems_of_accepted():
    item = dataset().items[0]
    answers = iter([reply(summary="Это хорошая новость."), reply()])
    retried = ev.evaluate_item(item, 1, lambda m: next(answers), None)
    first = ev.evaluate_item(item, 2, lambda m: reply(), None)
    summary = ev.summarize([retried, first])
    assert (summary.accepted, summary.first_try, summary.fallback, summary.errors) == (2, 1, 0, 0)
    assert summary.violations == {}


def test_no_data_text_names_the_data_directory():
    assert "data/news_eval/" in ev.NO_DATA_TEXT
