"""Тесты планировщика выдачи сводок на простых фейках.

Как test_market_tools.py для `run_tool_loop`: ни сети, ни Telegram, ни запуска процессов,
ни реального сна. Вызов инструмента сервера, отправка, часы и ожидание — внедряемые
фейки; на входе настоящие типы исключений python-telegram-bot, чтобы классификация
сбоев отправки проверялась по реальным классам.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from telegram.error import BadRequest, Forbidden, NetworkError

from price_watch.scheduler import (
    ACK_ATTEMPTS,
    RECHECK_SECONDS,
    TOOL_ACK,
    TOOL_RUN_DUE,
    TOOL_STOP,
    WatchScheduler,
    is_chat_unreachable,
    retry_pause,
    run_tick,
    split_message,
)

MSK = timezone(timedelta(hours=3))
T0 = datetime(2026, 9, 26, 12, 0, 0, tzinfo=MSK)


def _iso(minutes):
    return (T0 + timedelta(minutes=minutes)).isoformat()


def _report(report_id, chat_id, text="Сводка"):
    return {"report_id": report_id, "chat_id": chat_id, "text": text}


def _due(reports=(), next_due_minutes=None, has_more=False):
    return {
        "now": _iso(0),
        "next_due_at": None if next_due_minutes is None else _iso(next_due_minutes),
        "polled": {"chats": 0, "tickers": 0, "failed": 0},
        "reports": list(reports),
        "has_more": has_more,
    }


class FakeServer:
    """Фейковый сервер: ответы run_due по очереди (словарь или исключение), остальное записывает."""

    def __init__(self, *run_due_responses, ack_failures=0, stop_fails=False):
        self.run_due_responses = list(run_due_responses)
        self.calls = []
        self.ack_failures = ack_failures
        self.stop_fails = stop_fails

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == TOOL_RUN_DUE:
            response = self.run_due_responses.pop(0) if self.run_due_responses else _due()
            if isinstance(response, BaseException):
                raise response
            if callable(response):
                return response()
            return response
        if name == TOOL_ACK:
            if self.ack_failures > 0:
                self.ack_failures -= 1
                raise RuntimeError("ack failed")
            return {"acknowledged": len(arguments["report_ids"])}
        if name == TOOL_STOP:
            if self.stop_fails:
                raise RuntimeError("stop failed")
            return {"chat_id": arguments["chat_id"], "stopped": True}
        raise AssertionError(f"неожиданный вызов {name}")

    def called(self, name):
        return [arguments for called, arguments in self.calls if called == name]


class FakeSend:
    """Фейковая отправка: записывает (чат, текст); для указанных чатов бросает заданное исключение."""

    def __init__(self, failures=None):
        self.sent = []
        self.failures = failures or {}

    async def __call__(self, chat_id, text):
        if chat_id in self.failures:
            raise self.failures[chat_id]
        self.sent.append((chat_id, text))


# --- run_tick: доставка ---


def test_tick_delivers_each_report_to_its_chat_and_acks_them():
    server = FakeServer(_due([_report(1, 10, "для 10"), _report(2, 20, "для 20")], next_due_minutes=15))
    send = FakeSend()
    result = asyncio.run(run_tick(server.call_tool, send))
    assert send.sent == [(10, "для 10"), (20, "для 20")]
    assert server.called(TOOL_ACK) == [{"report_ids": [1, 2]}]
    assert server.called(TOOL_STOP) == []
    assert result.delivered == 2
    assert result.next_due_at == T0 + timedelta(minutes=15)
    assert result.repeat_now is False and result.retry_needed is False


def test_tick_without_reports_only_reads_the_next_due():
    server = FakeServer(_due(next_due_minutes=5))
    result = asyncio.run(run_tick(server.call_tool, FakeSend()))
    assert [name for name, _ in server.calls] == [TOOL_RUN_DUE]
    assert result.next_due_at == T0 + timedelta(minutes=5)
    assert result.delivered == 0


def test_tick_without_watches_has_no_next_due():
    result = asyncio.run(run_tick(FakeServer(_due()).call_tool, FakeSend()))
    assert result.next_due_at is None


def test_tick_unparseable_next_due_is_treated_as_missing():
    response = _due()
    response["next_due_at"] = "скоро"
    assert asyncio.run(run_tick(FakeServer(response).call_tool, FakeSend())).next_due_at is None


def test_tick_run_due_failure_propagates_to_the_caller():
    server = FakeServer(TimeoutError("late"))
    with pytest.raises(TimeoutError):
        asyncio.run(run_tick(server.call_tool, FakeSend()))


def test_long_report_is_sent_in_several_messages_without_loss():
    text = "\n".join(f"строка {i} " + "x" * 80 for i in range(120))
    server = FakeServer(_due([_report(1, 10, text)]))
    send = FakeSend()
    asyncio.run(run_tick(server.call_tool, send))
    assert len(send.sent) > 1
    assert all(len(chunk) <= 4000 for _, chunk in send.sent)
    assert "\n".join(chunk for _, chunk in send.sent) == text
    assert server.called(TOOL_ACK) == [{"report_ids": [1]}]


# --- run_tick: сбои доставки ---


def test_blocked_chat_is_stopped_and_other_chats_still_get_their_reports():
    server = FakeServer(_due([_report(1, 10), _report(2, 20)]))
    send = FakeSend({20: Forbidden("Forbidden: bot was blocked by the user")})
    result = asyncio.run(run_tick(server.call_tool, send))
    assert send.sent == [(10, "Сводка")]
    assert server.called(TOOL_ACK) == [{"report_ids": [1]}]  # подтверждена только доставленная
    assert server.called(TOOL_STOP) == [{"chat_id": 20}]
    assert result.stopped_chats == [20]
    assert result.retry_needed is False


def test_chat_not_found_counts_as_unreachable_but_other_bad_request_does_not():
    assert is_chat_unreachable(BadRequest("Chat not found")) is True
    assert is_chat_unreachable(BadRequest("Message is too long")) is False
    assert is_chat_unreachable(Forbidden("bot was blocked")) is True
    assert is_chat_unreachable(NetworkError("timeout")) is False
    assert is_chat_unreachable(RuntimeError("x")) is False


def test_transient_send_failure_leaves_the_report_unacked_and_asks_for_retry():
    server = FakeServer(_due([_report(1, 10)]))
    send = FakeSend({10: NetworkError("connection reset")})
    result = asyncio.run(run_tick(server.call_tool, send))
    assert server.called(TOOL_ACK) == []
    assert server.called(TOOL_STOP) == []
    assert result.retry_needed is True
    assert result.delivered == 0


def test_after_a_failure_later_reports_of_the_same_chat_are_held_back_to_keep_order():
    server = FakeServer(_due([_report(1, 10, "первая"), _report(2, 10, "вторая"), _report(3, 20, "чужая")]))
    send = FakeSend({10: NetworkError("boom")})
    asyncio.run(run_tick(server.call_tool, send))
    assert send.sent == [(20, "чужая")]
    assert server.called(TOOL_ACK) == [{"report_ids": [3]}]


def test_ack_is_retried_once_and_success_needs_no_retry():
    server = FakeServer(_due([_report(1, 10)]), ack_failures=1)
    result = asyncio.run(run_tick(server.call_tool, FakeSend()))
    assert len(server.called(TOOL_ACK)) == 2
    assert result.retry_needed is False


def test_ack_that_keeps_failing_asks_for_retry_instead_of_raising():
    server = FakeServer(_due([_report(1, 10)]), ack_failures=ACK_ATTEMPTS)
    result = asyncio.run(run_tick(server.call_tool, FakeSend()))
    assert len(server.called(TOOL_ACK)) == ACK_ATTEMPTS
    assert result.retry_needed is True


def test_failed_stop_of_an_unreachable_chat_asks_for_retry():
    server = FakeServer(_due([_report(1, 10)]), stop_fails=True)
    send = FakeSend({10: Forbidden("blocked")})
    result = asyncio.run(run_tick(server.call_tool, send))
    assert result.stopped_chats == []
    assert result.retry_needed is True


# --- run_tick: has_more ---


def test_has_more_with_progress_repeats_immediately():
    server = FakeServer(_due([_report(1, 10)], has_more=True))
    result = asyncio.run(run_tick(server.call_tool, FakeSend()))
    assert result.repeat_now is True and result.retry_needed is False


def test_has_more_without_progress_does_not_spin():
    server = FakeServer(_due([_report(1, 10)], has_more=True))
    result = asyncio.run(run_tick(server.call_tool, FakeSend({10: NetworkError("boom")})))
    assert result.repeat_now is False
    assert result.retry_needed is True


def test_has_more_with_only_a_stopped_chat_counts_as_progress():
    server = FakeServer(_due([_report(1, 10)], has_more=True))
    result = asyncio.run(run_tick(server.call_tool, FakeSend({10: Forbidden("blocked")})))
    assert result.repeat_now is True


# --- вспомогательные функции ---


def test_split_message_short_text_is_one_part():
    assert split_message("коротко", 100) == ["коротко"]


def test_split_message_breaks_on_line_boundaries_and_loses_nothing():
    text = "\n".join(["строка"] * 50)
    parts = split_message(text, 40)
    assert all(len(part) <= 40 for part in parts)
    assert "\n".join(parts) == text


def test_split_message_cuts_a_single_overlong_line():
    parts = split_message("x" * 250, 100)
    assert [len(part) for part in parts] == [100, 100, 50]


def test_split_message_text_of_exactly_the_limit_is_not_split():
    assert len(split_message("x" * 100, 100)) == 1


def test_retry_pause_doubles_up_to_the_cap():
    assert [retry_pause(n) for n in range(1, 8)] == [60, 120, 240, 480, 900, 900, 900]
    assert retry_pause(0) == 60


# --- WatchScheduler: цикл ---


class _Stop(Exception):
    """Ожидание фейка исчерпано: останавливает бесконечный цикл теста."""


class Clock:
    def __init__(self):
        self.moment = T0

    def now(self):
        return self.moment

    def advance(self, seconds):
        self.moment += timedelta(seconds=seconds)


class Script:
    """Фейковое ожидание: записывает запрошенные тайм-ауты и выполняет шаги по очереди.

    "elapse" — время прошло целиком (возврат False), ("wake", действие) — выполнить
    действие и вернуть True (разбудили раньше), когда шаги кончились — остановить цикл.
    """

    def __init__(self, clock, *steps):
        self.clock = clock
        self.steps = list(steps)
        self.timeouts = []

    async def __call__(self, timeout):
        self.timeouts.append(timeout)
        if not self.steps:
            raise _Stop
        step = self.steps.pop(0)
        if step == "elapse":
            self.clock.advance(timeout)
            return False
        _, action = step
        action()
        return True


def _run(scheduler):
    with pytest.raises(_Stop):
        asyncio.run(scheduler.run())


def _scheduler(server, script, clock, send=None):
    return WatchScheduler(server.call_tool, send or FakeSend(), now=clock.now, wait=script)


def test_first_tick_happens_at_once_before_any_waiting():
    clock, server = Clock(), FakeServer(_due())
    script = Script(clock)  # первый же вызов ожидания останавливает цикл
    _run(_scheduler(server, script, clock))
    assert [name for name, _ in server.calls] == [TOOL_RUN_DUE]
    assert script.timeouts == [RECHECK_SECONDS]  # сроков нет: ждём контрольной сверки


def test_waits_until_the_next_due_reported_by_the_server():
    clock, server = Clock(), FakeServer(_due(next_due_minutes=10), _due())
    script = Script(clock, "elapse")
    _run(_scheduler(server, script, clock))
    assert script.timeouts[0] == pytest.approx(600)
    assert len(server.called(TOOL_RUN_DUE)) == 2  # после ожидания сделан второй тик


def test_far_due_is_capped_by_the_recheck_period():
    clock, server = Clock(), FakeServer(_due(next_due_minutes=60 * 24), _due(next_due_minutes=60 * 24))
    script = Script(clock, "elapse")
    _run(_scheduler(server, script, clock))
    assert script.timeouts[0] == RECHECK_SECONDS
    assert len(server.called(TOOL_RUN_DUE)) == 2  # контрольная сверка вызывает выдачу


def test_schedule_at_earlier_moves_the_wake_up_forward():
    clock, server = Clock(), FakeServer(_due(next_due_minutes=60))
    holder = {}
    script = Script(clock, ("wake", lambda: holder["s"].schedule_at(T0 + timedelta(minutes=5))))
    holder["s"] = _scheduler(server, script, clock)
    _run(holder["s"])
    assert script.timeouts[0] == pytest.approx(3600)
    assert script.timeouts[1] == pytest.approx(300)  # перенесено на 5 минут


def test_schedule_at_later_is_ignored():
    clock, server = Clock(), FakeServer(_due(next_due_minutes=10))
    holder = {}
    script = Script(clock, ("wake", lambda: holder["s"].schedule_at(T0 + timedelta(minutes=50))))
    holder["s"] = _scheduler(server, script, clock)
    _run(holder["s"])
    assert script.timeouts[1] == pytest.approx(600)
    assert holder["s"].next_due == T0 + timedelta(minutes=10)


def test_schedule_at_when_nothing_is_planned_sets_the_first_due():
    clock, server = Clock(), FakeServer(_due())  # нет опросов: срока нет
    holder = {}
    script = Script(clock, ("wake", lambda: holder["s"].schedule_at(T0 + timedelta(minutes=15))))
    holder["s"] = _scheduler(server, script, clock)
    _run(holder["s"])
    assert script.timeouts[0] == RECHECK_SECONDS
    assert script.timeouts[1] == pytest.approx(900)


def test_schedule_at_during_a_tick_survives_the_tick_result():
    # /watch выполнился, пока шёл тик: ответ сервера получен ДО команды и о новом опросе
    # не знает (next_due_at = null). Подсказка не должна потеряться.
    clock = Clock()
    holder = {}

    def during_tick():
        holder["s"].schedule_at(T0 + timedelta(minutes=5))
        return _due()

    server = FakeServer(during_tick)
    script = Script(clock)
    holder["s"] = _scheduler(server, script, clock)
    _run(holder["s"])
    assert script.timeouts == [pytest.approx(300)]


def test_has_more_ticks_again_without_waiting():
    clock = Clock()
    server = FakeServer(_due([_report(1, 10)], has_more=True), _due([_report(2, 10)], next_due_minutes=15))
    script = Script(clock)
    _run(_scheduler(server, script, clock))
    assert len(server.called(TOOL_RUN_DUE)) == 2
    assert script.timeouts == [pytest.approx(900)]  # ни одного ожидания между тиками


def test_server_failures_back_off_and_success_resets_the_pause():
    clock = Clock()

    def success():
        # Срок считается от ТЕКУЩИХ часов теста: к этому моменту они ушли вперёд.
        response = _due()
        response["next_due_at"] = (clock.now() + timedelta(minutes=30)).isoformat()
        return response

    server = FakeServer(
        *[TimeoutError(str(n)) for n in range(6)],  # шесть неудач подряд
        success,
        TimeoutError("снова"),  # неудача после успеха
    )
    script = Script(clock, *["elapse"] * 7)
    scheduler = _scheduler(server, script, clock)
    _run(scheduler)
    assert script.timeouts[:6] == [60, 120, 240, 480, 900, 900]  # рост до предела
    assert script.timeouts[6] == pytest.approx(1800)  # успех: ждём срок сервера
    assert script.timeouts[7] == 60  # пауза после успеха началась заново


def test_failure_does_not_stop_the_scheduler_or_message_users():
    clock = Clock()
    server = FakeServer(RuntimeError("uv not found"), _due())
    send = FakeSend()
    script = Script(clock, "elapse")
    _run(_scheduler(server, script, clock, send))
    assert len(server.called(TOOL_RUN_DUE)) == 2  # после сбоя цикл продолжил работу
    assert send.sent == []  # пользователям о фоновых сбоях не пишем


def test_transient_delivery_failure_retries_after_the_growing_pause():
    clock = Clock()
    server = FakeServer(
        _due([_report(1, 10)], next_due_minutes=5),
        _due([_report(1, 10)], next_due_minutes=5),
    )
    send = FakeSend({10: NetworkError("boom")})
    script = Script(clock, "elapse", "elapse")
    _run(_scheduler(server, script, clock, send))
    assert script.timeouts[:2] == [60, 120]  # повтор не чаще раза в минуту и с ростом паузы


def test_successful_delivery_after_failures_resets_the_pause():
    clock = Clock()
    server = FakeServer(
        _due([_report(1, 10)]),
        _due([_report(1, 10)], next_due_minutes=30),
    )
    flaky = {"fail": True}

    class Send:
        sent = []

        async def __call__(self, chat_id, text):
            if flaky["fail"]:
                flaky["fail"] = False
                raise NetworkError("boom")
            self.sent.append((chat_id, text))

    send = Send()
    script = Script(clock, "elapse")
    _run(_scheduler(server, script, clock, send))
    assert script.timeouts[0] == 60
    assert send.sent == [(10, "Сводка")]


def test_first_tick_after_restart_delivers_overdue_reports_once():
    clock = Clock()
    server = FakeServer(_due([_report(7, 10, "за простой")], next_due_minutes=60))
    send = FakeSend()
    _run(_scheduler(server, Script(clock), clock, send))
    assert send.sent == [(10, "за простой")]
    assert server.called(TOOL_ACK) == [{"report_ids": [7]}]


# --- WatchScheduler: реальное ожидание и отмена ---


def test_real_wait_wakes_on_schedule_at_and_cancellation_ends_the_task_cleanly():
    async def scenario():
        server = FakeServer(_due(), _due(), _due())
        scheduler = WatchScheduler(server.call_tool, FakeSend())  # настоящие часы и ожидание
        task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.05)
        assert len(server.called(TOOL_RUN_DUE)) == 1  # первый тик сразу при запуске
        scheduler.schedule_at(datetime.now(timezone.utc))  # «уже пора» будит цикл
        await asyncio.sleep(0.05)
        assert len(server.called(TOOL_RUN_DUE)) == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
