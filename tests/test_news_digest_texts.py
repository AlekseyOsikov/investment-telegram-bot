"""Тесты текстов дайджеста новостей (news_digest/texts.py): оговорки в каждом сообщении,
отсутствие оценок и гарантий в фиксированных текстах, деление длинного дайджеста."""

import re

from news_digest import texts

ANALYSIS = {
    "event_type": "dividends",
    "importance": "high",
    "summary": "Совет директоров рекомендовал выплатить дивиденды.",
    "mechanism": "Дивидендная отсечка уменьшает стоимость акции на размер выплаты.",
}


def item(n=1, analysis=ANALYSIS):
    return {
        "id": f"id{n}",
        "source": "example.com",
        "link": f"https://example.com/{n}",
        "title": f"Альфатрейд объявила дивиденды {n}",
        "published_at": "2026-10-18T09:00:00+00:00",
        "tickers": [{"ticker": "ALFA", "matched": "Альфатрейд", "kind": "name"}],
        "analysis": analysis,
    }


def assert_has_disclaimers(text):
    for part in (
        texts.DISCLAIMER_ADVISOR,
        texts.DISCLAIMER_NOT_ADVICE,
        texts.DISCLAIMER_COVERAGE,
        texts.DISCLAIMER_AUTO,
    ):
        assert part in text


def test_enabled_reply_has_disclaimers_and_limits():
    text = texts.news_enabled_text(["ALFA", "BETA"], 6, 6, uncovered=["BETA"])
    assert_has_disclaimers(text)
    assert "ALFA, BETA" in text and "раз в 6 часов" in text and "6 сообщений" in text
    assert "Покрытие ограничено по бумагам BETA" in text


def test_enabled_reply_without_uncovered_has_no_limited_note():
    assert "Покрытие ограничено" not in texts.news_enabled_text(["ALFA"], 1, 3)
    assert "раз в час" in texts.news_enabled_text(["ALFA"], 1, 3)


def test_digest_message_has_disclaimers_analysis_and_source():
    (message,) = texts.digest_messages([item()], {"id1": "Цена в это же время: +1,2%"})
    assert_has_disclaimers(message)
    assert "Тип события: Дивиденды · важность: высокая" in message
    assert "https://example.com/1" in message and "[ALFA]" in message
    assert "Цена в это же время: +1,2%" in message


def test_unavailable_analysis_is_marked():
    (message,) = texts.digest_messages([item(analysis=None)])
    assert texts.UNAVAILABLE_NOTE in message and "Тип события" not in message
    assert_has_disclaimers(message)


def test_empty_digest_is_not_sent():
    assert texts.digest_messages([]) == []


def test_long_digest_is_split_and_every_part_has_disclaimers():
    messages = texts.digest_messages([item(n) for n in range(30)])
    assert len(messages) > 1
    for message in messages:
        assert len(message) <= 4000
        assert_has_disclaimers(message)
    assert "часть 1 из" in messages[0]
    shown = "".join(messages)
    assert all(f"https://example.com/{n}" in shown for n in range(30))


def test_urgent_message_has_disclaimers():
    message = texts.urgent_message(item())
    assert_has_disclaimers(message)
    assert "Важная новость" in message


def test_status_and_stop_texts():
    on = texts.news_status_text(True, ["ALFA"], "2026-10-18T12:00:00+03:00", 2, 6)
    assert "18.10 12:00 (МСК)" in on and "2 из 6" in on
    assert texts.DISCLAIMER_ADVISOR in on
    assert "выключен" in texts.news_status_text(False, [], None, 0, 6)
    assert "удалены" in texts.news_stop_text(True)
    assert "не был включён" in texts.news_stop_text(False)
    assert "/watch" in texts.NEED_WATCH_TEXT


# В фиксированных текстах не должно быть оценок, гарантий и обещаний доходности.
GUARANTEE = re.compile(
    r"гарантир\w+|точно\s+(?:вырастет|упадёт)|наверняка|обязательно\s+(?:вырастет|принесёт)|"
    r"доходность\s+составит|хорошая\s+новость|плохая\s+новость|позитивн\w+|негативн\w+",
    re.IGNORECASE,
)


def test_fixed_texts_have_no_guarantees_or_assessments():
    fixed = [
        texts.DISCLAIMERS,
        texts.UNAVAILABLE_NOTE,
        texts.NEED_WATCH_TEXT,
        texts.NEWS_FAILED_TEXT,
        texts.news_enabled_text(["ALFA"], 6, 6, ["BETA"]),
        texts.news_status_text(True, ["ALFA"], None, 0, 6, ["BETA"]),
        texts.news_stop_text(True),
        texts.digest_messages([item()])[0],
        texts.urgent_message(item()),
    ]
    for text in fixed:
        assert GUARANTEE.search(text) is None, text
