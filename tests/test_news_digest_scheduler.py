"""Тесты тика и цикла планировщика дайджеста новостей (news_digest/scheduler.py).

Сервер новостей — НАСТОЯЩИЙ `NewsService` на временной БД (контракт инструментов проверяется
вместе с ботом), ленты и сервер опроса цен, разбор и отправка — фейки; часы управляются тестом.
Без сети, процессов и сна. Компании вымышленные.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import news_rules as nr
import news_server as srv
import news_store as ns
import pytest
from telegram.error import Forbidden

from news_digest import analyzer, scheduler, texts

START = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)
TABLE = nr.parse_alias_table({"ALFA": {"names": ["Альфатрейд"]}, "BETA": {"names": ["Бетагаз"]}})
GOOD = {
    "event_type": "dividends",
    "importance": "low",
    "summary": "Совет директоров рекомендовал выплату.",
    "mechanism": "Дивидендная отсечка уменьшает стоимость акции на размер выплаты.",
}


class Clock:
    def __init__(self):
        self.now = START

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


def rss(*entries):
    """entries: (заголовок, ссылка, момент публикации)."""
    items = "".join(
        f"<item><title>{t}</title><link>{link}</link><pubDate>{format_datetime(p)}</pubDate></item>"
        for t, link, p in entries
    )
    return f"<rss><channel>{items}</channel></rss>"


class World:
    """Окружение теста: сервер новостей, опросы цен, разбор, отправка."""

    def __init__(self, tmp_path, bodies=None):
        self.clock = Clock()
        self.bodies = bodies if bodies is not None else {}
        self.entries = {}
        self.service = srv.NewsService(
            ns.NewsStore(tmp_path / "news.sqlite3"),
            list(self.bodies),
            TABLE,
            7,
            lambda url: self.bodies[url],
            self.clock,
        )
        self.watch = {}  # chat_id -> тикеры (None — опроса нет)
        self.watch_errors = set()
        self.sent = []  # (chat_id, text)
        self.send_errors = {}  # chat_id -> исключение
        self.model = lambda news: analyzer.AnalysisOutcome(dict(GOOD), 1)
        self.analyzed_ids = []
        self.price = None
        self.ack_log = []

    async def call_news(self, name, args=None):
        args = args or {}
        if name == "news_ack":
            self.ack_log.append((name, args["chat_id"], list(args["news_ids"]), args["kind"]))
        method = {
            "news_chats": lambda: self.service.chats(),
            "news_collect": lambda: self.service.collect(args.get("tickers")),
            "news_pending": lambda: self.service.pending(args["tickers"], args.get("limit", 20)),
            "news_save_analysis": lambda: self.service.save_analysis(
                args["news_id"],
                args.get("available", True),
                args.get("event_type", ""),
                args.get("importance", ""),
                args.get("summary", ""),
                args.get("mechanism", ""),
            ),
            "news_state": lambda: self.service.state(args["chat_id"]),
            "news_ready": lambda: self.service.ready(
                args["chat_id"], args["tickers"], args.get("limit", 20)
            ),
            "news_ack": lambda: self.service.ack(args["chat_id"], args["news_ids"], args["kind"]),
            "news_disable": lambda: self.service.disable(args["chat_id"]),
            "news_enable": lambda: self.service.enable(args["chat_id"]),
        }[name]
        return method()

    async def call_watch(self, name, args=None):
        chat_id = args["chat_id"]
        if chat_id in self.watch_errors:
            raise TimeoutError("нет ответа")
        tickers = self.watch.get(chat_id)
        return {"active": True, "secids": tickers} if tickers else {"active": False}

    async def analyze(self, news):
        self.analyzed_ids.append(news["id"])
        return self.model(news)

    async def send(self, chat_id, text):
        error = self.send_errors.get(chat_id)
        if error:
            raise error
        self.sent.append((chat_id, text))

    async def price_line(self, item):
        return self.price

    def deps(self):
        return scheduler.Deps(
            self.call_watch, self.call_news, self.analyze, self.send, self.clock, self.price_line
        )

    async def tick(self, **settings):
        base = dict(
            collect_interval_minutes=15,
            digest_interval_hours=6,
            digest_max_items=10,
            max_messages_per_day=6,
            max_analyze_per_tick=5,
        )
        return await scheduler.run_tick(self.deps(), scheduler.Settings(**{**base, **settings}))

    def enable(self, chat_id, tickers=("ALFA",)):
        self.watch[chat_id] = list(tickers)
        self.service.enable(chat_id)

    def publish(self, *titles, source="https://a.example/rss", age_hours=1):
        published = self.clock.now - timedelta(hours=age_hours)
        entries = self.entries.setdefault(source, [])
        entries += [(t, f"https://a.example/{abs(hash(t))}", published) for t in titles]
        self.bodies[source] = rss(*entries)
        if source not in self.service.feeds:
            self.service.feeds.append(source)


def run(coro):
    return asyncio.run(coro)


def test_no_chats_means_no_work(tmp_path):
    async def scenario():
        world = World(tmp_path)
        result = await world.tick()
        assert not result.retry_needed and world.sent == [] and world.analyzed_ids == []

    run(scenario())


def test_digest_not_sent_before_interval_then_sent_with_disclaimers(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish("Альфатрейд объявила дивиденды")
        await world.tick()
        assert world.sent == []  # разобрано, но срок дайджеста ещё не наступил
        assert len(world.analyzed_ids) == 1
        world.clock.advance(hours=6, minutes=1)
        result = await world.tick()
        assert result.digests == 1 and len(world.sent) == 1
        chat_id, text = world.sent[0]
        assert chat_id == 1 and texts.DISCLAIMER_NOT_ADVICE in text
        assert "Альфатрейд объявила дивиденды" in text and "Дивиденды" in text
        assert world.ack_log[-1][3] == "digest"

    run(scenario())


def test_empty_digest_is_not_sent_and_nothing_resent(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.clock.advance(hours=7)
        await world.tick()
        assert world.sent == []
        world.publish("Альфатрейд объявила дивиденды")
        await world.tick()
        await world.tick()
        assert len(world.sent) == 1  # вторая отправка той же новости не случилась

    run(scenario())


def test_news_from_two_feeds_is_sent_once(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        published = world.clock.now - timedelta(hours=1)
        world.bodies["https://a.example/rss"] = rss(
            ("Альфатрейд объявила дивиденды", "https://x.example/1?utm_a=1", published)
        )
        world.bodies["https://b.example/rss"] = rss(
            ("Альфатрейд объявила дивиденды", "https://x.example/1", published)
        )
        world.service.feeds[:] = list(world.bodies)
        world.clock.advance(hours=7)
        await world.tick()
        assert len(world.analyzed_ids) == 1
        assert world.sent[0][1].count("Альфатрейд объявила дивиденды") == 1

    run(scenario())


def test_send_failure_leaves_news_unconfirmed_and_retries(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish("Альфатрейд объявила дивиденды")
        world.clock.advance(hours=7)
        world.send_errors[1] = RuntimeError("сеть")
        result = await world.tick()
        assert result.retry_needed and result.digests == 0
        assert world.ack_log == []  # подтверждения не было
        del world.send_errors[1]
        result = await world.tick()
        assert result.digests == 1 and len(world.sent) == 1

    run(scenario())


def test_ack_happens_only_after_successful_send(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish("Альфатрейд объявила дивиденды")
        world.clock.advance(hours=7)
        events = []
        original_send, original_ack = world.send, world.call_news

        async def send(chat_id, text):
            events.append("send")
            await original_send(chat_id, text)

        async def call_news(name, args=None):
            if name == "news_ack":
                events.append("ack")
            return await original_ack(name, args)

        world.send, world.call_news = send, call_news
        await world.tick()
        assert events == ["send", "ack"]

    run(scenario())


def test_urgent_goes_immediately_before_digest_time(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.model = lambda news: analyzer.AnalysisOutcome({**GOOD, "importance": "high"}, 1)
        world.publish("Альфатрейд объявила дивиденды")
        result = await world.tick()
        assert result.urgent == 1 and len(world.sent) == 1
        assert "Важная новость" in world.sent[0][1] and world.ack_log[0][3] == "urgent"
        world.clock.advance(hours=7)
        await world.tick()
        assert len(world.sent) == 1  # та же новость в дайджест не попала

    run(scenario())


def test_urgent_precedes_digest_and_digest_keeps_one_slot_of_the_limit(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        urgent_titles = [f"Альфатрейд срочная {i}" for i in range(4)]
        world.model = lambda news: analyzer.AnalysisOutcome(
            {**GOOD, "importance": "high" if "срочная" in news["title"] else "low"}, 1
        )
        world.publish(*urgent_titles, "Альфатрейд обычная новость")
        world.clock.advance(hours=7)
        await world.tick(max_messages_per_day=3)
        kinds = [entry[3] for entry in world.ack_log]
        assert kinds == ["urgent", "urgent", "digest"]  # два срочных и дайджест = лимит 3
        assert "срочная" in world.sent[0][1] and "обычная" in world.sent[2][1]

    run(scenario())


def test_daily_limit_blocks_and_next_day_delivers(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        settings = dict(max_messages_per_day=1, digest_interval_hours=1)
        world.publish("Альфатрейд раз")
        world.clock.advance(hours=2)
        await world.tick(**settings)
        assert len(world.sent) == 1
        world.publish("Альфатрейд два")
        world.clock.advance(hours=2)
        await world.tick(**settings)
        assert len(world.sent) == 1  # лимит суток исчерпан, новость ждёт, а не теряется
        world.clock.advance(hours=24)
        world.publish("Альфатрейд три")
        await world.tick(**settings)
        assert len(world.sent) == 2 and "Альфатрейд два" in world.sent[1][1]

    run(scenario())


def test_digest_max_items_leaves_the_rest_for_next_digest(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish(*[f"Альфатрейд новость {i}" for i in range(4)])
        world.clock.advance(hours=7)
        await world.tick(digest_max_items=3, max_analyze_per_tick=10)
        assert world.sent[0][1].count("Альфатрейд новость") == 3
        world.clock.advance(hours=7)
        await world.tick(digest_max_items=3)
        assert "Альфатрейд новость" in world.sent[1][1]

    run(scenario())


def test_stopped_watch_disables_digest(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.watch[1] = None  # /watch_stop
        result = await world.tick()
        assert result.disabled_chats == [1] and world.service.state(1)["enabled"] is False
        assert world.sent == []

    run(scenario())


def test_watch_error_skips_chat_without_disabling(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.watch_errors.add(1)
        result = await world.tick()
        assert result.retry_needed and world.service.state(1)["enabled"] is True

    run(scenario())


def test_unreachable_chat_is_disabled_others_unaffected(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.enable(2)
        world.publish("Альфатрейд объявила дивиденды")
        world.clock.advance(hours=7)
        world.send_errors[1] = Forbidden("bot was blocked by the user")
        result = await world.tick()
        assert result.disabled_chats == [1]
        assert world.service.state(1)["enabled"] is False
        assert [chat for chat, _ in world.sent] == [2]

    run(scenario())


def test_chats_see_only_their_tickers(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1, ["ALFA"])
        world.enable(2, ["BETA"])
        world.publish("Альфатрейд раз", "Бетагаз два")
        world.clock.advance(hours=7)
        await world.tick()
        by_chat = dict(world.sent)
        assert "Альфатрейд раз" in by_chat[1] and "Бетагаз" not in by_chat[1]
        assert "Бетагаз два" in by_chat[2] and "Альфатрейд" not in by_chat[2]

    run(scenario())


def test_model_down_keeps_fresh_news_pending_then_shows_without_analysis(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish("Альфатрейд объявила дивиденды", age_hours=1)
        world.model = lambda news: analyzer.AnalysisOutcome(
            None, 1, "модель недоступна", transient=True
        )
        result = await world.tick(max_analyze_per_tick=5)
        assert result.retry_needed and len(world.analyzed_ids) == 1  # остальные не мучаем
        assert world.service.pending(["ALFA"])["items"] != []  # не сохранено как «без разбора»
        world.clock.advance(hours=7)  # новость стала старше срока ожидания разбора
        await world.tick()
        assert texts.UNAVAILABLE_NOTE in world.sent[0][1]

    run(scenario())


def test_final_violation_is_saved_as_no_analysis_and_shown_with_note(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish("Альфатрейд объявила дивиденды")
        world.model = lambda news: analyzer.AnalysisOutcome(None, 2, "запрещённые формулировки")
        world.clock.advance(hours=7)
        await world.tick()
        assert texts.UNAVAILABLE_NOTE in world.sent[0][1]
        assert world.service.pending(["ALFA"])["items"] == []

    run(scenario())


def test_price_line_is_shown_when_available(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish("Альфатрейд объявила дивиденды")
        world.clock.advance(hours=7)
        world.price = "Цена в это же время: +1%"
        await world.tick()
        assert "Цена в это же время: +1%" in world.sent[0][1]

    run(scenario())


def test_price_line_failure_does_not_block_the_digest(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish("Альфатрейд объявила дивиденды")
        world.clock.advance(hours=7)

        async def broken(item):
            raise RuntimeError("биржа недоступна")

        world.price_line = broken
        await world.tick()
        assert world.sent and "Цена в это же время" not in world.sent[0][1]

    run(scenario())


def test_collect_failure_marks_retry_but_delivers_what_is_stored(tmp_path):
    async def scenario():
        world = World(tmp_path)
        world.enable(1)
        world.publish("Альфатрейд объявила дивиденды")
        await world.tick()
        world.clock.advance(hours=7)
        original = world.call_news

        async def call_news(name, args=None):
            if name == "news_collect":
                raise TimeoutError("ленты")
            return await original(name, args)

        world.call_news = call_news
        result = await world.tick()
        assert result.retry_needed and result.digests == 1

    run(scenario())


def test_news_server_failure_propagates_to_scheduler(tmp_path):
    async def scenario():
        world = World(tmp_path)

        async def broken(name, args=None):
            raise FileNotFoundError("python")

        world.call_news = broken
        with pytest.raises(FileNotFoundError):
            await world.tick()

    run(scenario())


# --- цикл планировщика --------------------------------------------------------------------


def make_scheduler(world, waits):
    async def wait(timeout):
        waits.append(timeout)
        return False

    return scheduler.NewsScheduler(
        world.deps(), scheduler.Settings(collect_interval_minutes=15), wait=wait
    )


def test_scheduler_waits_collect_interval_after_success(tmp_path):
    async def scenario():
        world = World(tmp_path)
        waits = []
        sched = make_scheduler(world, waits)
        await sched.tick()
        assert sched.next_due == world.clock.now + timedelta(minutes=15)

    run(scenario())


def test_scheduler_backs_off_with_doubling_pause_and_resets(tmp_path):
    async def scenario():
        world = World(tmp_path)
        sched = make_scheduler(world, [])
        good = world.call_news

        async def broken(name, args=None):
            raise TimeoutError("сервер")

        world.call_news = broken
        sched._deps.call_news = broken
        await sched.tick()
        assert sched.next_due == world.clock.now + timedelta(seconds=60)
        await sched.tick()
        assert sched.next_due == world.clock.now + timedelta(seconds=120)
        sched._deps.call_news = good
        await sched.tick()
        assert sched.next_due == world.clock.now + timedelta(minutes=15)
        sched._deps.call_news = broken
        await sched.tick()
        assert sched.next_due == world.clock.now + timedelta(seconds=60)  # счётчик сбросился

    run(scenario())


def test_request_tick_makes_next_tick_immediate(tmp_path):
    async def scenario():
        world = World(tmp_path)
        sched = make_scheduler(world, [])
        await sched.tick()
        sched.request_tick()
        assert sched.next_due == world.clock.now

    run(scenario())
