"""Тесты разбора новости под слотом очереди (news_digest/analyzer.py) на фейках модели и шлюза."""

import asyncio
import json
from types import SimpleNamespace

from llm_queue.gate import LlmGate
from llm_queue.rules import INTERACTIVE, KIND_MAIN, KIND_NEWS
from news_digest import analyzer, rules

NEWS = {
    "id": "n1",
    "source": "example.com",
    "link": "https://example.com/1",
    "title": "Альфатрейд объявила дивиденды",
    "excerpt": "",
    "published_at": "2026-10-18T09:00:00+00:00",
    "tickers": [{"ticker": "ALFA", "matched": "Альфатрейд", "kind": "name"}],
}
GOOD = json.dumps(
    {
        "event_type": "dividends",
        "importance": "high",
        "summary": "Совет директоров рекомендовал выплатить дивиденды.",
        "mechanism": "Дивидендная отсечка уменьшает стоимость акции на размер выплаты.",
    },
    ensure_ascii=False,
)
BAD = GOOD.replace("Совет директоров рекомендовал выплатить дивиденды.", "Акция вырастет.")


def make_gate():
    return LlmGate(1, 1, 10, 1800, timeout_seconds=600)


def run(coro):
    return asyncio.run(coro)


class Script:
    """Фейк модели: отдаёт ответы по очереди, запоминает сообщения."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_correct_answer_is_accepted_on_first_attempt():
    async def scenario():
        model = Script(GOOD)
        outcome = await analyzer.analyze_news(NEWS, make_gate(), model, "ОСНОВА")
        assert outcome.analysis["event_type"] == "dividends" and outcome.attempts == 1
        assert model.calls[0][0]["content"] == "ОСНОВА"

    run(scenario())


def test_violation_then_retry_then_success():
    async def scenario():
        model = Script(BAD, GOOD)
        outcome = await analyzer.analyze_news(NEWS, make_gate(), model, None)
        assert outcome.analysis is not None and outcome.attempts == 2
        retry_user_message = model.calls[1][-1]["content"]
        assert "нарушил правила" in retry_user_message and "запрещённые" in retry_user_message

    run(scenario())


def test_violation_persists_falls_back_to_no_analysis():
    async def scenario():
        model = Script(BAD, BAD)
        outcome = await analyzer.analyze_news(NEWS, make_gate(), model, None)
        assert outcome.analysis is None and outcome.attempts == rules.MAX_ATTEMPTS
        assert "запрещённые" in outcome.reason
        assert model.replies == []  # третьего обращения нет

    run(scenario())


def test_unavailable_model_gives_no_analysis_without_retry_and_frees_slot():
    async def scenario():
        gate = make_gate()
        model = Script(ConnectionError("нет связи"))
        outcome = await analyzer.analyze_news(NEWS, gate, model, None)
        assert outcome.analysis is None and outcome.attempts == 1
        assert "ConnectionError" in outcome.reason
        # слот свободен: следующий вопрос пользователя выполняется сразу
        done = []

        async def work(ctx):
            done.append(1)

        sub = gate.submit(5, KIND_MAIN, "вопрос", work)
        await sub.job.task
        assert done == [1]

    run(scenario())


def test_analysis_is_invisible_to_user_queue_and_limits():
    async def scenario():
        gate = make_gate()
        seen = {}

        def model(messages):
            seen["chat_jobs_during"] = gate.chat_jobs(0)
            return GOOD

        await analyzer.analyze_news(NEWS, gate, model, None)
        assert seen["chat_jobs_during"] == []
        assert gate.user_jobs_done == 0
        assert gate.chat_jobs(0) == []
        assert gate.stats.count(KIND_NEWS) == 1  # длительность учтена по своему виду

    run(scenario())


def test_user_question_gets_slot_between_two_analyses():
    async def scenario():
        gate = make_gate()
        order = []
        first_started = asyncio.Event()
        release = asyncio.Event()

        def first_model(messages):
            order.append("news-1")
            return GOOD

        async def slow_first():
            async with gate.slot(priority=1, kind=KIND_NEWS) as ctx:
                first_started.set()
                await release.wait()
                await ctx.run_thread(first_model, [])

        async def second_analysis():
            order.append("news-2-start")
            await analyzer.analyze_news(NEWS, gate, lambda m: order.append("news-2") or GOOD, None)

        async def user_work(ctx):
            order.append("user")

        first = asyncio.create_task(slow_first())
        await first_started.wait()
        second = asyncio.create_task(second_analysis())
        await asyncio.sleep(0)
        sub = gate.submit(
            7, KIND_MAIN, "вопрос", user_work
        )  # приходит, пока первая новость держит слот
        release.set()
        await asyncio.gather(first, second, sub.job.task)
        assert order.index("user") < order.index("news-2")

    run(scenario())


def test_cancellation_is_propagated():
    async def scenario():
        gate = make_gate()
        started = asyncio.Event()

        def model(messages):
            return GOOD

        async def blocker():
            async with gate.slot(priority=INTERACTIVE, kind=KIND_MAIN):
                started.set()
                await asyncio.sleep(10)

        hold = asyncio.create_task(blocker())
        await started.wait()
        task = asyncio.create_task(analyzer.analyze_news(NEWS, gate, model, None))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()
        hold.cancel()
        try:
            await hold
        except asyncio.CancelledError:
            pass

    run(scenario())


# --- вызов модели ---------------------------------------------------------------------------


class FakeClient:
    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_with and "response_format" in kwargs:
            raise self.fail_with
        message = SimpleNamespace(content=GOOD)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def backend(client, provider="ollama", model="m"):
    return SimpleNamespace(client=client, model=model, provider=provider)


def test_local_model_gets_no_cloud_parameters():
    client = FakeClient()
    text = analyzer.make_completer(backend(client), 30)([{"role": "user", "content": "x"}])
    assert text == GOOD
    sent = client.calls[0]
    assert "extra_body" not in sent and sent["temperature"] == 0 and sent["timeout"] == 30
    assert sent["response_format"] == {"type": "json_object"}


def test_cloud_model_disables_thinking_and_kimi_has_no_temperature():
    deepseek = FakeClient()
    analyzer.make_completer(backend(deepseek, "deepseek"), 30)([])
    assert deepseek.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    kimi = FakeClient()
    analyzer.make_completer(backend(kimi, "kimi"), 30)([])
    assert "temperature" not in kimi.calls[0]


def test_json_mode_rejected_by_server_retries_without_it():
    class Bad400(Exception):
        status_code = 400

    client = FakeClient(fail_with=Bad400("response_format"))
    assert analyzer.make_completer(backend(client), 30)([]) == GOOD
    assert len(client.calls) == 2 and "response_format" not in client.calls[1]


def test_other_errors_propagate():
    class Down(Exception):
        status_code = 503

    client = FakeClient(fail_with=Down())
    try:
        analyzer.make_completer(backend(client), 30)([])
    except Down:
        pass
    else:
        raise AssertionError("ошибка 503 не должна проглатываться")


def test_zero_attempts_gives_exhausted_outcome_without_calling_the_model(monkeypatch):
    monkeypatch.setattr(rules, "MAX_ATTEMPTS", 0)

    async def scenario():
        model = Script(GOOD)
        outcome = await analyzer.analyze_news(NEWS, make_gate(), model, None)
        assert outcome.analysis is None
        assert outcome.attempts == 0
        assert outcome.reason == "исчерпаны попытки"
        assert model.calls == []

    run(scenario())
