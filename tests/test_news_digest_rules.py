"""Тесты чистых правил дайджеста новостей (news_digest/rules.py): проверка ответа модели,
запретные формулировки, повтор/откат, выбор элементов, сроки. Компании вымышленные."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from news_digest import rules

LINK = "https://example.com/news/1"
GOOD = {
    "event_type": "dividends",
    "importance": "high",
    "summary": "Совет директоров рекомендовал выплатить дивиденды за год.",
    "mechanism": "Дивидендная отсечка уменьшает стоимость акции на размер выплаты.",
}


def reply(**changes):
    return json.dumps({**GOOD, **changes}, ensure_ascii=False)


def test_correct_answer_passes():
    result = rules.validate_analysis(reply(), LINK)
    assert result.ok and result.analysis.event_type == "dividends"
    assert result.analysis.importance == "high" and result.problems == ()


def test_json_wrapped_in_text_or_fence_is_parsed():
    wrapped = "Вот разбор:\n```json\n" + reply() + "\n```\nГотово."
    assert rules.validate_analysis(wrapped, LINK).ok
    inline = "Ответ: " + reply() + " Спасибо!"
    assert rules.validate_analysis(inline, LINK).ok


@pytest.mark.parametrize("raw", [None, "", "не json", "[1, 2]", "{broken"])
def test_not_a_json_object_is_rejected(raw):
    result = rules.validate_analysis(raw, LINK)
    assert not result.ok and "JSON" in result.problems[0]


def test_enums_and_empty_fields_checked():
    assert not rules.validate_analysis(reply(event_type="rumor"), LINK).ok
    assert not rules.validate_analysis(reply(importance="critical"), LINK).ok
    assert not rules.validate_analysis(reply(summary="  "), LINK).ok
    assert not rules.validate_analysis(reply(mechanism=None), LINK).ok
    assert rules.validate_analysis(reply(event_type=" Dividends ", importance="HIGH"), LINK).ok


def test_length_limits_and_missing_link():
    assert not rules.validate_analysis(reply(summary="а" * 301), LINK).ok
    assert not rules.validate_analysis(reply(mechanism="б" * 301), LINK).ok
    result = rules.validate_analysis(reply(), "  ")
    assert not result.ok and "ссылк" in result.problems[0]


# Каждая запретная формулировка из набора границы должна отвергаться в ЛЮБОМ из полей.
FORBIDDEN_SAMPLES = [
    "Новость позитивна для акций.",
    "Событие негативно для компании.",
    "Акция вырастет после публикации.",
    "Бумага упадёт на этом фоне.",
    "Ожидается рост котировок.",
    "Это хорошая новость для держателей.",
    "Это плохая новость для акционеров.",
    "Акции подорожают, а облигации подешевеют.",
    "Происходит падение цены бумаги.",
    "Стоит купить акции до отсечки.",
    "Лучше продавать бумагу сейчас.",
    "Рекомендуем держать позицию.",
    "Мы ожидаем улучшения.",
    "Это сигнал к покупке.",
    "Прогноз цены улучшился.",
    "Скорее всего, интерес вырастет.",
    "Событие приведёт к пересмотру оценок.",
    "Новость повлияет на цену.",
    "Бычий сценарий реализуется.",
    "Медвежий сентимент усилится.",
    "Целевая цена повышена.",
    "Гарантированная выплата.",
]


@pytest.mark.parametrize("text", FORBIDDEN_SAMPLES)
@pytest.mark.parametrize("field", ["summary", "mechanism"])
def test_forbidden_wording_is_rejected_everywhere(text, field):
    result = rules.validate_analysis(reply(**{field: text}), LINK)
    assert not result.ok, text
    assert any("запрещённые" in problem and field in problem for problem in result.problems)


@pytest.mark.parametrize(
    "text",
    [
        "Ростелеком опубликовал отчёт по МСФО.",
        "Дивидендная отсечка уменьшает стоимость акции на размер выплаты.",
        "Ограничение по ценной бумаге влияет на условия торгов.",
        "Совет директоров рекомендовал выплатить дивиденды.",
        "Смена руководителя может менять оценку управленческих рисков.",
    ],
)
def test_neutral_wording_is_not_flagged(text):
    assert rules.find_forbidden(text) == []


def test_forbidden_list_is_not_trivially_short():
    # каждый шаблон перечня хоть раз сработал бы на собственной основе слова — шаблон не пустой
    assert len(rules.FORBIDDEN_PATTERNS) >= 30
    assert all(pattern.strip() for pattern in rules.FORBIDDEN_PATTERNS)


def test_next_step_accept_retry_fallback():
    bad = rules.validate_analysis("мусор", LINK)
    good = rules.validate_analysis(reply(), LINK)
    assert rules.next_step(1, good) == rules.ACCEPT
    assert rules.next_step(1, bad) == rules.RETRY
    assert rules.next_step(2, bad) == rules.FALLBACK
    assert rules.next_step(2, good) == rules.ACCEPT


def item(link, importance=None, published="2026-10-18T10:00:00+00:00"):
    analysis = None if importance is None else {"importance": importance}
    return {"link": link, "analysis": analysis, "published_at": published}


def test_sort_by_importance_then_freshness_with_unanalyzed_last():
    items = [
        item("low-old", "low", "2026-10-18T08:00:00+00:00"),
        item("none", None, "2026-10-18T12:00:00+00:00"),
        item("high", "high", "2026-10-18T07:00:00+00:00"),
        item("low-new", "low", "2026-10-18T11:00:00+00:00"),
        item("med", "medium"),
    ]
    assert [i["link"] for i in rules.sort_items(items)] == [
        "high",
        "med",
        "low-new",
        "low-old",
        "none",
    ]


def test_select_digest_limits_and_keeps_rest_for_next_time():
    items = [item(f"n{i}", "low") for i in range(5)]
    assert len(rules.select_digest(items, 3)) == 3
    assert rules.select_digest(items, 0) == []


def test_urgent_only_high_analyzed_and_within_budget():
    items = [item("a", "high"), item("b", "high"), item("c", "medium"), item("d", None)]
    assert {i["link"] for i in rules.select_urgent(items, 5)} == {"a", "b"}
    assert len(rules.select_urgent(items, 1)) == 1
    assert rules.select_urgent(items, 0) == []
    assert rules.is_urgent(item("x", "high")) and not rules.is_urgent(item("y", None))


def test_messages_left_never_negative():
    assert rules.messages_left(2, 6) == 4
    assert rules.messages_left(9, 6) == 0


def test_moments_and_next_digest():
    assert rules.parse_moment("2026-10-18T10:00:00+03:00").utcoffset() == timedelta(hours=3)
    assert rules.parse_moment("2026-10-18T10:00:00").tzinfo == timezone.utc
    assert rules.parse_moment("мусор") is None and rules.parse_moment(None) is None
    enabled = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
    assert rules.next_digest_at(enabled, None, 6) == enabled + timedelta(hours=6)
    last = enabled + timedelta(hours=7)
    assert rules.next_digest_at(enabled, last, 6) == last + timedelta(hours=6)


# --- Дополнение: терпимый разбор JSON -----------------------------------------


def test_parse_json_object_cases():
    assert rules.parse_json_object(None) is None
    assert rules.parse_json_object("") is None
    assert rules.parse_json_object('{"a": 1}') == {"a": 1}
    assert rules.parse_json_object('Вот ответ: {"a": 1} Спасибо.') == {"a": 1}
    assert rules.parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    assert rules.parse_json_object('```\n{"a": 3}\n```') == {"a": 3}


@pytest.mark.parametrize(
    "raw", ["[1, 2]", '"строка"', "42", "{обрезанный", "текст } без { пары"]
)
def test_parse_json_object_of_non_object_or_broken_text_is_none(raw):
    assert rules.parse_json_object(raw) is None


def test_parse_json_object_skips_broken_braces_before_a_real_object():
    assert rules.parse_json_object('{не json} затем {"a": 1}') == {"a": 1}


def test_parse_json_object_prefers_fenced_block_over_surrounding_text():
    raw = 'Сначала {"a": 1}\n```json\n{"b": 2}\n```'
    assert rules.parse_json_object(raw) == {"b": 2}
