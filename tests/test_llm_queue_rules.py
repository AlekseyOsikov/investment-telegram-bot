"""Чистые правила очереди LLM-задач (llm_queue/rules.py): без сети и без asyncio."""

import re

import pytest

from llm_queue import rules
from llm_queue.rules import (
    INTERACTIVE,
    KIND_MAIN,
    KIND_RESEARCH,
    KIND_SMART_AGENT,
    RESEARCH,
    DurationStats,
    Estimate,
    QueueEntry,
    QueueItemView,
)


def entry(seq, chat=1, kind=KIND_MAIN, priority=INTERACTIVE, running=False):
    return QueueEntry(seq=seq, chat_id=chat, kind=kind, priority=priority, running=running)


# --- порядок и позиция ---


def test_interactive_overtakes_research():
    research = entry(1, kind=KIND_RESEARCH, priority=RESEARCH)
    user = entry(2)
    assert [e.seq for e in rules.order_waiting([research, user])] == [2, 1]


def test_fifo_within_priority():
    a, b, c = entry(5), entry(3), entry(4)
    assert [e.seq for e in rules.order_waiting([a, b, c])] == [3, 4, 5]


def test_running_not_in_waiting_order():
    run = entry(1, running=True)
    wait = entry(2)
    assert rules.order_waiting([run, wait]) == [wait]


def test_position_counts_running_and_waiting_ahead():
    run = entry(1, running=True)
    w1, w2 = entry(2, chat=2), entry(3, chat=3)
    entries = [run, w1, w2]
    assert rules.queue_position(entries, run) == 0
    assert rules.queue_position(entries, w1) == 2
    assert rules.queue_position(entries, w2) == 3


def test_position_after_removing_other_task():
    run = entry(1, running=True)
    w1, w2 = entry(2, chat=2), entry(3, chat=3)
    assert rules.queue_position([run, w1, w2], w2) == 3
    assert rules.queue_position([run, w2], w2) == 2


def test_interactive_ahead_of_earlier_research():
    research = entry(1, kind=KIND_RESEARCH, priority=RESEARCH)
    user = entry(2)
    assert rules.queue_position([research, user], user) == 1
    assert rules.queue_position([research, user], research) == 2


# --- статистика и оценка ---


def filled(kind, values):
    stats = DurationStats()
    for v in values:
        stats.record(kind, v)
    return stats


def test_median_and_p90():
    stats = filled(KIND_MAIN, [10, 20, 30, 40, 50])
    assert stats.median(KIND_MAIN) == 30
    assert stats.p90(KIND_MAIN) == 50


def test_window_keeps_last_20():
    stats = DurationStats()
    for i in range(30):
        stats.record(KIND_MAIN, float(i))
    assert stats.count(KIND_MAIN) == 20
    assert stats.median(KIND_MAIN) == 19.5  # значения 10..29


def test_negative_duration_ignored():
    stats = DurationStats()
    stats.record(KIND_MAIN, -1.0)
    assert stats.count(KIND_MAIN) == 0


def test_estimate_few_samples_has_no_low_and_uses_timeout():
    stats = filled(KIND_MAIN, [60, 60])  # меньше MIN_SAMPLES
    est = rules.estimate_wait(stats, [], KIND_MAIN, 1, timeout_seconds=600)
    assert est == Estimate(low_minutes=None, high_minutes=10)


def test_estimate_no_data_at_all():
    est = rules.estimate_wait(DurationStats(), [KIND_MAIN], KIND_MAIN, 1, timeout_seconds=120)
    assert est.low_minutes is None
    assert est.high_minutes == 4  # две задачи по таймауту 120 с


def test_estimate_k1_sums_all_ahead_plus_own():
    stats = filled(KIND_MAIN, [60, 60, 60, 120])  # медиана 60, p90 120
    est = rules.estimate_wait(stats, [KIND_MAIN, KIND_MAIN], KIND_MAIN, 1, 600)
    assert est == Estimate(low_minutes=3, high_minutes=6)


def test_estimate_k2_halves_waves():
    stats = filled(KIND_MAIN, [60, 60, 60, 120])
    est = rules.estimate_wait(stats, [KIND_MAIN, KIND_MAIN, KIND_MAIN], KIND_MAIN, 2, 600)
    # 3 впереди при K=2 → 2 волны + своя = 3 задачи
    assert est == Estimate(low_minutes=3, high_minutes=6)


def test_estimate_mixed_kinds_uses_each_kind_stats():
    stats = DurationStats()
    for v in (60, 60, 60):
        stats.record(KIND_MAIN, v)
    for v in (300, 300, 300):
        stats.record(KIND_SMART_AGENT, v)
    est = rules.estimate_wait(stats, [KIND_SMART_AGENT], KIND_MAIN, 1, 600)
    assert est == Estimate(low_minutes=6, high_minutes=6)


def test_estimate_rounds_up_to_minute():
    stats = filled(KIND_MAIN, [61, 61, 61])
    est = rules.estimate_wait(stats, [], KIND_MAIN, 1, 600)
    assert est.low_minutes == 2 and est.high_minutes == 2


def test_estimate_one_kind_unknown_drops_low_only():
    stats = filled(KIND_MAIN, [60, 60, 60])
    est = rules.estimate_wait(stats, [KIND_SMART_AGENT], KIND_MAIN, 1, 300)
    assert est.low_minutes is None
    assert est.high_minutes == 6  # 300 с (таймаут) + 60 с (p90 main)


# --- лимиты ---


def test_limits_boundaries():
    assert not rules.chat_limit_reached(0, 1)
    assert rules.chat_limit_reached(1, 1)
    assert rules.chat_limit_reached(3, 2)
    assert not rules.queue_full(9, 10)
    assert rules.queue_full(10, 10)
    assert not rules.ttl_expired(1800, 1800)
    assert rules.ttl_expired(1800.1, 1800)


# --- разбор /queue_cancel ---


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], rules.CancelArg(all_jobs=True)),
        (["  "], rules.CancelArg(all_jobs=True)),
        (["2"], rules.CancelArg(number=2)),
        (["0"], rules.CancelArg(number=0)),
        (["-1"], rules.CancelArg(number=-1)),
        (["abc"], rules.CancelArg(error=True)),
        (["1", "2"], rules.CancelArg(error=True)),
        (["1_0"], rules.CancelArg(error=True)),
        (["2.5"], rules.CancelArg(error=True)),
    ],
)
def test_parse_cancel_arg(args, expected):
    assert rules.parse_cancel_arg(args) == expected


def test_select_by_number():
    jobs = ["a", "b"]
    assert rules.select_by_number(jobs, 1) == "a"
    assert rules.select_by_number(jobs, 2) == "b"
    assert rules.select_by_number(jobs, 0) is None
    assert rules.select_by_number(jobs, -1) is None
    assert rules.select_by_number(jobs, 3) is None


# --- тексты ---


def all_texts():
    est = Estimate(2, 5)
    return [
        rules.text_accepted(3, est),
        rules.text_accepted(3, Estimate(None, 5)),
        rules.text_chat_limit(0, None),
        rules.text_chat_limit(2, est),
        rules.text_busy(),
        rules.text_expired(),
        rules.text_cancel_all(2),
        rules.text_cancel_one(2, running=False),
        rules.text_cancel_one(1, running=True),
        rules.text_nothing_to_cancel(),
        rules.text_bad_number(),
        rules.text_already_saved(),
        rules.format_queue_list(
            [
                QueueItemView(
                    1,
                    0,
                    None,
                    "Что такое облигации",
                )
            ]
        ),
    ]


def test_accepted_text_is_approximate_with_cancel_command():
    text = rules.text_accepted(2, Estimate(2, 5))
    assert "примерно 2–5 мин" in text
    assert "/queue_cancel" in text
    assert "/cancel " not in text


def test_estimate_without_low_bound():
    assert rules.format_estimate(Estimate(None, 7)) == "примерно до 7 мин"
    assert rules.format_estimate(Estimate(3, 3)) == "примерно 3 мин"


def test_texts_have_no_promises_or_money():
    forbidden = re.compile(r"гаранти|обещаем|точно через|₽|руб|\$|usd|доход", re.IGNORECASE)
    for text in all_texts():
        assert not forbidden.search(text), text


def test_queue_list_numbers_and_truncation():
    long_text = "один два три четыре пять шесть семь восемь девять десять"
    items = [
        QueueItemView(1, 0, None, "Короткий вопрос"),
        QueueItemView(2, 3, Estimate(2, 5), long_text),
    ]
    text = rules.format_queue_list(items)
    lines = text.splitlines()
    assert lines[1].startswith("1. Короткий вопрос — выполняется")
    assert lines[2].startswith("2. один два три четыре пять шесть… — позиция 3")
    assert "примерно 2–5 мин" in lines[2]
    assert "семь" not in text


def test_queue_list_empty():
    assert "нет" in rules.format_queue_list([])


def test_snippet_char_limit():
    text = "а" * 200
    result = rules.snippet(text)
    assert result.endswith("…")
    assert len(result) <= rules.SNIPPET_MAX_CHARS + 1


# --- кооперативная отмена вопроса агента ---


def test_raise_if_cancelled():
    rules.raise_if_cancelled(None)
    rules.raise_if_cancelled(lambda: False)
    with pytest.raises(rules.AskCancelled):
        rules.raise_if_cancelled(lambda: True)
