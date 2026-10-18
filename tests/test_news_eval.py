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
