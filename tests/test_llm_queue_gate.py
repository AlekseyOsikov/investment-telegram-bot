"""Шлюз очереди LLM-задач (llm_queue/gate.py) на фиктивных корутинах: без сети и Telegram."""

import asyncio
import threading

from llm_queue.gate import LlmGate
from llm_queue.rules import (
    KIND_MAIN,
    KIND_SMART_AGENT,
    RESEARCH,
)


def make_gate(concurrency=1, per_user=1, queue_max=10, ttl=1800):
    return LlmGate(concurrency, per_user, queue_max, ttl, timeout_seconds=600)


def run(coro):
    return asyncio.run(coro)


async def settle(times=5):
    for _ in range(times):
        await asyncio.sleep(0)


def test_free_slot_runs_immediately_without_status():
    async def scenario():
        gate = make_gate()
        done = []

        async def work(ctx):
            done.append(1)

        sub = gate.submit(1, KIND_MAIN, "q", work)
        assert sub.status == "accepted" and not sub.queued
        await sub.job.task
        assert done == [1]
        assert gate.chat_jobs(1) == []
        assert gate.user_jobs_done == 1
        assert gate.stats.count(KIND_MAIN) == 1

    run(scenario())


def test_second_chat_waits_and_runs_in_order():
    async def scenario():
        gate = make_gate()
        order = []
        release = asyncio.Event()

        async def first(ctx):
            order.append("a-start")
            await release.wait()
            order.append("a-end")

        async def second(ctx):
            order.append("b")

        a = gate.submit(1, KIND_MAIN, "a", first)
        await settle()
        b = gate.submit(2, KIND_MAIN, "b", second)
        assert b.queued and b.position == 2
        assert order == ["a-start"]
        release.set()
        await b.job.task
        assert order == ["a-start", "a-end", "b"]
        await a.job.task

    run(scenario())


def test_chat_limit_and_busy():
    async def scenario():
        gate = make_gate(per_user=1, queue_max=1)
        block = asyncio.Event()

        async def work(ctx):
            await block.wait()

        a = gate.submit(1, KIND_MAIN, "a", work)
        await settle()
        again = gate.submit(1, KIND_MAIN, "a2", work)
        assert again.status == "chat_limit" and again.position == 0
        b = gate.submit(2, KIND_MAIN, "b", work)  # ждёт — глубина 1
        assert b.status == "accepted"
        c = gate.submit(3, KIND_MAIN, "c", work)
        assert c.status == "busy"
        block.set()
        await asyncio.gather(a.job.task, b.job.task)

    run(scenario())


def test_same_chat_questions_never_overlap_with_concurrency_two():
    async def scenario():
        gate = make_gate(concurrency=2, per_user=2)
        active = 0
        peak = 0

        async def work(ctx):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

        a = gate.submit(1, KIND_MAIN, "a", work)
        b = gate.submit(1, KIND_MAIN, "b", work)
        await asyncio.gather(a.job.task, b.job.task)
        assert peak == 1

    run(scenario())


def test_interactive_overtakes_research_slot_waiter():
    async def scenario():
        gate = make_gate()
        order = []
        hold = asyncio.Event()

        async def blocker(ctx):
            await hold.wait()

        async def research_step():
            async with gate.slot(RESEARCH):
                order.append("research")

        async def user(ctx):
            order.append("user")

        gate.submit(1, KIND_MAIN, "block", blocker)
        await settle()
        r = asyncio.create_task(research_step())
        await settle()
        u = gate.submit(2, KIND_MAIN, "u", user)
        # Дать задаче пользователя встать в ожидание слота ДО освобождения: wait_for на 3.11
        # запускает внутреннюю задачу (на 3.12+ — нет), и без этого порядок зависит от версии.
        await settle()
        hold.set()
        await asyncio.gather(r, u.job.task)
        assert order == ["user", "research"]

    run(scenario())


def test_cancel_waiting_job_removes_it_and_never_runs():
    async def scenario():
        gate = make_gate(per_user=2)
        ran = []
        block = asyncio.Event()

        async def blocker(ctx):
            await block.wait()

        async def victim(ctx):
            ran.append("victim")

        gate.submit(9, KIND_MAIN, "block", blocker)
        await settle()
        sub = gate.submit(1, KIND_MAIN, "v", victim)
        assert len(gate.chat_jobs(1)) == 1
        cancelled = gate.cancel_chat(1, "test")
        assert [(j.seq, running) for j, running in cancelled] == [(sub.job.seq, False)]
        assert gate.chat_jobs(1) == []
        block.set()
        await settle(10)
        assert ran == []
        assert gate.view_chat(1) == []

    run(scenario())


def test_cancel_one_of_two_keeps_other_and_renumbers():
    async def scenario():
        gate = make_gate(per_user=2)
        block = asyncio.Event()

        async def work(ctx):
            await block.wait()

        a = gate.submit(1, KIND_MAIN, "first", work)
        await settle()
        b = gate.submit(1, KIND_MAIN, "second", work)
        view = gate.view_chat(1)
        assert [(v.number, v.position) for v in view] == [(1, 0), (2, 2)]
        gate.cancel_job(b.job, "user")
        view = gate.view_chat(1)
        assert [(v.number, v.text) for v in view] == [(1, "first")]
        block.set()
        await a.job.task

    run(scenario())


def test_cancel_running_async_job_frees_slot_immediately():
    async def scenario():
        gate = make_gate()
        started = asyncio.Event()
        ran_next = []

        async def long(ctx):
            started.set()
            await asyncio.sleep(3600)

        async def nxt(ctx):
            ran_next.append(1)

        a = gate.submit(1, KIND_MAIN, "long", long)
        await started.wait()
        b = gate.submit(2, KIND_MAIN, "next", nxt)
        gate.cancel_job(a.job, "user")
        await b.job.task
        assert ran_next == [1]
        assert gate.stats.count(KIND_MAIN) == 1  # снятый не попал в статистику

    run(scenario())


def test_cancel_running_thread_defers_slot_release_until_thread_ends():
    async def scenario():
        gate = make_gate()
        thread_started = threading.Event()
        thread_release = threading.Event()
        late_calls = []
        ran_next = []

        def blocking():
            thread_started.set()
            thread_release.wait(5)
            return "saved"

        async def late(result, exc):
            late_calls.append((result, exc))

        async def agent(ctx):
            await ctx.run_thread(blocking, late=late)

        async def nxt(ctx):
            ran_next.append(1)

        a = gate.submit(1, KIND_SMART_AGENT, "q", agent)
        while not thread_started.is_set():
            await asyncio.sleep(0.005)
        b = gate.submit(2, KIND_MAIN, "n", nxt)
        gate.cancel_job(a.job, "user")
        await settle(10)
        assert ran_next == []  # слот не освобождён: поток ещё работает
        assert gate.chat_jobs(1) == []
        thread_release.set()
        await b.job.task
        assert ran_next == [1]
        assert late_calls == [("saved", None)]

    run(scenario())


def test_ttl_expires_waiting_interactive_job_with_notice():
    async def scenario():
        gate = make_gate(ttl=0.05)
        block = asyncio.Event()
        notices = []
        ran = []

        async def blocker(ctx):
            await block.wait()

        async def notify(text):
            notices.append(text)

        async def victim(ctx):
            ran.append(1)

        a = gate.submit(1, KIND_MAIN, "block", blocker)
        await settle()
        b = gate.submit(2, KIND_MAIN, "v", victim, notify=notify)
        await b.job.task
        assert len(notices) == 1 and "очереди" in notices[0]
        assert ran == [] and gate.chat_jobs(2) == []
        block.set()
        await a.job.task

    run(scenario())


def test_research_priority_has_no_ttl():
    async def scenario():
        gate = make_gate(ttl=0.02)
        block = asyncio.Event()
        ran = []

        async def blocker(ctx):
            await block.wait()

        async def research(ctx):
            ran.append(1)

        gate.submit(1, KIND_MAIN, "b", blocker)
        await settle()
        r = gate.submit(2, "research", "r", research, priority=RESEARCH)
        await asyncio.sleep(0.1)
        assert ran == []
        block.set()
        await r.job.task
        assert ran == [1]

    run(scenario())


def test_slot_context_counts_waiting_and_user_questions():
    async def scenario():
        gate = make_gate()
        block = asyncio.Event()

        async def blocker(ctx):
            await block.wait()

        a = gate.submit(1, KIND_MAIN, "b", blocker)
        await settle()
        before = gate.user_jobs_done

        async def step():
            async with gate.slot(RESEARCH) as ctx:
                return ctx.waited_seconds

        t = asyncio.create_task(step())
        await asyncio.sleep(0.05)
        block.set()
        waited = await t
        await a.job.task
        assert waited >= 0.04
        assert gate.user_jobs_done == before + 1

    run(scenario())


def test_cancelling_slot_waiter_does_not_leak_slot():
    async def scenario():
        gate = make_gate()
        block = asyncio.Event()

        async def blocker(ctx):
            await block.wait()

        a = gate.submit(1, KIND_MAIN, "b", blocker)
        await settle()

        async def step():
            async with gate.slot(RESEARCH):
                pass

        t = asyncio.create_task(step())
        await settle()
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        block.set()
        await a.job.task
        # слот свободен: новая задача стартует сразу
        done = []

        async def work(ctx):
            done.append(1)

        b = gate.submit(2, KIND_MAIN, "w", work)
        assert not b.queued
        await b.job.task
        assert done == [1]

    run(scenario())


def test_shutdown_cancels_everything():
    async def scenario():
        gate = make_gate(per_user=2)

        async def forever(ctx):
            await asyncio.sleep(3600)

        gate.submit(1, KIND_MAIN, "a", forever)
        await settle()
        gate.submit(2, KIND_MAIN, "b", forever)
        await gate.shutdown()
        assert gate.chat_jobs(1) == [] and gate.chat_jobs(2) == []

    run(scenario())


# --- Дополнение: признаки отмены, сбои уведомлений, гонка слота и остановка ---------------------

import logging  # noqa: E402

from llm_queue.gate import Job  # noqa: E402


def test_job_context_exposes_cancel_flag_to_work_and_threads():
    async def scenario():
        gate = make_gate()
        seen = []
        release = asyncio.Event()

        async def work(ctx):
            seen.append((ctx.cancelled, ctx.should_cancel()))
            await release.wait()
            seen.append((ctx.cancelled, ctx.should_cancel()))

        sub = gate.submit(1, KIND_MAIN, "q", work)
        await settle()
        sub.job.cancelled = True  # как после cancel_job: флаг видят и корутина, и поток
        release.set()
        await sub.job.task
        assert seen == [(False, False), (True, True)]

    run(scenario())


def test_waited_seconds_of_job_is_visible_in_context():
    async def scenario():
        gate = make_gate()
        waits = []

        async def work(ctx):
            waits.append(ctx.waited_seconds)

        sub = gate.submit(1, KIND_MAIN, "q", work)
        await sub.job.task
        assert len(waits) == 1 and waits[0] >= 0.0

    run(scenario())


def test_late_callback_receives_exception_of_cancelled_thread():
    async def scenario():
        gate = make_gate()
        started = threading.Event()
        release = threading.Event()
        late_calls = []

        def failing():
            started.set()
            release.wait(5)
            raise ValueError("поток упал")

        async def late(result, error):
            late_calls.append((result, error))

        async def work(ctx):
            await ctx.run_thread(failing, late=late)

        sub = gate.submit(1, KIND_SMART_AGENT, "q", work)
        while not started.is_set():
            await asyncio.sleep(0.005)
        gate.cancel_job(sub.job, "user")
        await settle()
        release.set()
        for _ in range(100):
            if late_calls:
                break
            await asyncio.sleep(0.01)
        assert len(late_calls) == 1
        result, error = late_calls[0]
        assert result is None
        assert isinstance(error, ValueError) and str(error) == "поток упал"

    run(scenario())


def test_failing_late_callback_is_logged_and_slot_is_still_released(caplog):
    async def scenario():
        gate = make_gate()
        started = threading.Event()
        release = threading.Event()
        ran_next = []

        def blocking():
            started.set()
            release.wait(5)
            return "ok"

        async def late(result, error):
            raise RuntimeError("не удалось уведомить")

        async def agent(ctx):
            await ctx.run_thread(blocking, late=late)

        async def nxt(ctx):
            ran_next.append(1)

        a = gate.submit(1, KIND_SMART_AGENT, "q", agent)
        while not started.is_set():
            await asyncio.sleep(0.005)
        b = gate.submit(2, KIND_MAIN, "n", nxt)
        gate.cancel_job(a.job, "user")
        await settle()
        release.set()
        await b.job.task
        await settle(10)
        assert ran_next == [1]

    with caplog.at_level(logging.ERROR):
        run(scenario())
    assert "Сбой уведомления после отмены вопроса" in caplog.text


def test_expired_job_without_notify_callback_is_dropped_silently():
    async def scenario():
        gate = make_gate(ttl=0.05)
        block = asyncio.Event()
        ran = []

        async def blocker(ctx):
            await block.wait()

        async def victim(ctx):
            ran.append(1)

        a = gate.submit(1, KIND_MAIN, "block", blocker)
        await settle()
        b = gate.submit(2, KIND_MAIN, "v", victim)
        await b.job.task
        assert b.job.expired is True
        assert ran == [] and gate.chat_jobs(2) == []
        block.set()
        await a.job.task

    run(scenario())


def test_failing_expiry_notice_is_logged_and_does_not_break_the_gate(caplog):
    async def scenario():
        gate = make_gate(ttl=0.05)
        block = asyncio.Event()

        async def blocker(ctx):
            await block.wait()

        async def notify(text):
            raise ConnectionError("чат недоступен")

        async def victim(ctx):
            raise AssertionError("не должен запускаться")

        a = gate.submit(1, KIND_MAIN, "block", blocker)
        await settle()
        b = gate.submit(2, KIND_MAIN, "v", victim, notify=notify)
        await b.job.task
        assert b.job.expired is True
        block.set()
        await a.job.task
        assert gate.chat_jobs(1) == [] and gate.chat_jobs(2) == []

    with caplog.at_level(logging.WARNING):
        run(scenario())
    assert "Не удалось отправить служебное сообщение очереди" in caplog.text


def test_slot_handed_over_but_waiter_cancelled_in_same_tick_does_not_leak_slot():
    async def scenario():
        gate = make_gate()
        holder_ready = asyncio.Event()
        holder_release = asyncio.Event()

        async def holder():
            async with gate.slot(RESEARCH):
                holder_ready.set()
                await holder_release.wait()

        async def waiter():
            async with gate.slot(RESEARCH):
                raise AssertionError("слот не должен достаться отменённому")

        holder_task = asyncio.create_task(holder())
        await holder_ready.wait()
        waiter_task = asyncio.create_task(waiter())
        await settle()
        holder_release.set()
        await asyncio.sleep(0)  # holder выходит и выдаёт слот ожидающему, тот ещё не проснулся
        assert gate._free == 0 and not gate._waiters
        waiter_task.cancel()
        await asyncio.gather(holder_task, waiter_task, return_exceptions=True)
        await settle()

        done = []

        async def work(ctx):
            done.append(1)

        sub = gate.submit(1, KIND_MAIN, "w", work)
        assert not sub.queued  # слот свободен, утечки нет
        await sub.job.task
        assert done == [1]

    run(scenario())


def test_shutdown_cancels_unregistered_job_with_task_and_waits_for_it():
    async def scenario():
        gate = make_gate()
        job = Job(
            seq=99,
            chat_id=0,
            kind="research",
            priority=RESEARCH,
            text="",
            enqueued_at=0.0,
            registered=False,
        )
        job.task = asyncio.create_task(asyncio.sleep(3600))
        gate._jobs[job.seq] = job
        await gate.shutdown()
        assert job.task.cancelled()

    run(scenario())


def test_shutdown_with_nothing_running_returns_at_once():
    async def scenario():
        await make_gate().shutdown()

    run(scenario())
