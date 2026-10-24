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


# --------------------------------------------------------------------------- #
# Дополнение: вспомогательные шаги тика, цикл ожидания, строка о цене, остановка
# --------------------------------------------------------------------------- #

import logging  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from price_watch.texts import MSK  # noqa: E402


def fk_deps(call_news, call_watch=None, analyze=None, send=None, now=None, price_line=None):
    async def no_watch(name, args=None):
        return {}

    async def no_analyze(news):
        return analyzer.AnalysisOutcome(None, 0, "нет")

    async def no_send(chat_id, text):
        return None

    return scheduler.Deps(
        call_watch or no_watch,
        call_news,
        analyze or no_analyze,
        send or no_send,
        now or (lambda: START),
        price_line,
    )


def test_price_line_helper_is_none_without_provider():
    async def scenario():
        deps = fk_deps(None)
        assert await scheduler._price_line(deps, {"id": "n1"}) is None

    run(scenario())


def test_price_line_helper_returns_the_line_and_swallows_failures():
    async def scenario():
        async def good(item):
            return f"строка {item['id']}"

        async def broken(item):
            raise RuntimeError("сервер цен недоступен")

        assert await scheduler._price_line(fk_deps(None, price_line=good), {"id": "n1"}) == (
            "строка n1"
        )
        assert await scheduler._price_line(fk_deps(None, price_line=broken), {"id": "n1"}) is None

    run(scenario())


def test_disable_chat_records_chat_and_calls_server():
    async def scenario():
        calls = []

        async def call_news(name, args=None):
            calls.append((name, args))
            return {}

        result = scheduler.TickResult()
        await scheduler._disable_chat(fk_deps(call_news), 7, result, "причина")
        assert calls == [("news_disable", {"chat_id": 7})]
        assert result.disabled_chats == [7]
        assert result.retry_needed is False

    run(scenario())


def test_disable_chat_failure_asks_for_retry_and_does_not_record_chat():
    async def scenario():
        async def call_news(name, args=None):
            raise TimeoutError("сервер")

        result = scheduler.TickResult()
        await scheduler._disable_chat(fk_deps(call_news), 7, result, "причина")
        assert result.disabled_chats == []
        assert result.retry_needed is True

    run(scenario())


def test_analyze_pending_save_failure_marks_retry_and_continues_with_next_news():
    async def scenario():
        saved = []

        async def call_news(name, args=None):
            if name == "news_pending":
                return {"items": [{"id": "n1"}, {"id": "n2"}]}
            if args["news_id"] == "n1":
                raise TimeoutError("сервер")
            saved.append(args)
            return {}

        async def analyze(news):
            return analyzer.AnalysisOutcome(dict(GOOD), 1)

        result = scheduler.TickResult()
        await scheduler._analyze_pending(
            fk_deps(call_news, analyze=analyze), scheduler.Settings(), ["ALFA"], result
        )
        assert result.retry_needed is True
        assert result.analyzed == 1
        assert [args["news_id"] for args in saved] == ["n2"]
        assert saved[0]["available"] is True
        assert saved[0]["event_type"] == GOOD["event_type"]

    run(scenario())


def test_analyze_pending_without_items_does_nothing():
    async def scenario():
        async def call_news(name, args=None):
            return {"items": None}

        result = scheduler.TickResult()
        await scheduler._analyze_pending(fk_deps(call_news), scheduler.Settings(), [], result)
        assert (result.analyzed, result.retry_needed) == (0, False)

    run(scenario())


def test_analyze_pending_stale_transient_failure_is_saved_without_analysis():
    async def scenario():
        saved = []
        old = (START - timedelta(hours=scheduler.ANALYSIS_GRACE_HOURS + 1)).isoformat()

        async def call_news(name, args=None):
            if name == "news_pending":
                return {"items": [{"id": "n1", "published_at": old}]}
            saved.append(args)
            return {}

        async def analyze(news):
            return analyzer.AnalysisOutcome(None, 0, "модель недоступна", transient=True)

        result = scheduler.TickResult()
        await scheduler._analyze_pending(
            fk_deps(call_news, analyze=analyze), scheduler.Settings(), [], result
        )
        assert saved == [{"news_id": "n1", "available": False}]
        assert result.retry_needed is False
        assert result.analyzed == 1

    run(scenario())


def test_deliver_chat_of_disabled_chat_does_not_ask_for_news():
    async def scenario():
        asked = []

        async def call_news(name, args=None):
            asked.append(name)
            return {"enabled": False}

        result = scheduler.TickResult()
        await scheduler._deliver_chat(
            fk_deps(call_news), scheduler.Settings(), 1, ["ALFA"], result
        )
        assert asked == ["news_state"]
        assert (result.digests, result.urgent) == (0, 0)

    run(scenario())


def test_run_tick_logs_unavailable_feeds_without_failing(caplog):
    async def scenario():
        async def call_news(name, args=None):
            return {
                "news_chats": {"chat_ids": [1]},
                "news_collect": {
                    "feeds_unavailable": [{"url": "https://x.example", "error": "404"}]
                },
                "news_pending": {"items": []},
                "news_state": {"enabled": False},
            }[name]

        async def call_watch(name, args=None):
            return {"active": True, "secids": ["alfa"]}

        with caplog.at_level(logging.WARNING):
            result = await scheduler.run_tick(fk_deps(call_news, call_watch))
        assert result.retry_needed is False

    run(scenario())
    assert "https://x.example" in caplog.text
    assert "404" in caplog.text


def test_run_tick_analysis_failure_marks_retry_and_still_delivers():
    async def scenario():
        asked = []

        async def call_news(name, args=None):
            asked.append(name)
            if name == "news_pending":
                raise ValueError("сбой разбора")
            return {
                "news_chats": {"chat_ids": [1]},
                "news_collect": {},
                "news_state": {"enabled": False},
            }[name]

        async def call_watch(name, args=None):
            return {"active": True, "secids": ["ALFA"]}

        result = await scheduler.run_tick(fk_deps(call_news, call_watch))
        assert result.retry_needed is True
        assert "news_state" in asked

    run(scenario())


def test_run_tick_chat_failure_marks_retry_and_other_chats_still_processed():
    async def scenario():
        states = []

        async def call_news(name, args=None):
            if name == "news_state":
                states.append(args["chat_id"])
                if args["chat_id"] == 1:
                    raise ValueError("сбой чата")
                return {"enabled": False}
            return {
                "news_chats": {"chat_ids": [1, 2]},
                "news_collect": {},
                "news_pending": {"items": []},
            }[name]

        async def call_watch(name, args=None):
            return {"active": True, "secids": ["ALFA"]}

        result = await scheduler.run_tick(fk_deps(call_news, call_watch))
        assert result.retry_needed is True
        assert states == [1, 2]

    run(scenario())


def test_run_tick_propagates_cancellation_from_analysis_and_delivery():
    async def scenario(failing):
        async def call_news(name, args=None):
            if name == failing:
                raise asyncio.CancelledError
            return {
                "news_chats": {"chat_ids": [1]},
                "news_collect": {},
                "news_pending": {"items": []},
                "news_state": {"enabled": False},
            }[name]

        async def call_watch(name, args=None):
            return {"active": True, "secids": ["ALFA"]}

        with pytest.raises(asyncio.CancelledError):
            await scheduler.run_tick(fk_deps(call_news, call_watch))

    run(scenario("news_pending"))
    run(scenario("news_state"))


def test_wait_for_wake_times_out_with_false_and_returns_true_when_woken():
    async def scenario():
        sched = scheduler.NewsScheduler(fk_deps(None))
        assert await sched._wait_for_wake(0.01) is False
        sched._wake.set()
        assert await sched._wait_for_wake(5) is True

    run(scenario())


def test_seconds_until_due_cases():
    now = [START]
    sched = scheduler.NewsScheduler(
        fk_deps(None, now=lambda: now[0]), scheduler.Settings(collect_interval_minutes=15)
    )
    assert sched._seconds_until_due() == 900.0  # срок ещё не назначен
    sched._due = START + timedelta(seconds=90)
    assert sched._seconds_until_due() == 90.0
    now[0] = START + timedelta(seconds=100)
    assert sched._seconds_until_due() == 0.0  # срок прошёл


def test_run_loop_ticks_when_due_waits_otherwise_and_skips_tick_after_wake():
    async def scenario():
        ticks = []
        waits = []
        answers = iter([True, False])

        async def call_news(name, args=None):
            ticks.append(name)
            return {"chat_ids": []}

        async def wait(timeout):
            waits.append(timeout)
            try:
                return next(answers)
            except StopIteration:
                raise asyncio.CancelledError from None

        sched = scheduler.NewsScheduler(
            fk_deps(call_news), scheduler.Settings(collect_interval_minutes=15), wait=wait
        )
        with pytest.raises(asyncio.CancelledError):
            await sched.run()
        # Первый тик — сразу (срок = сейчас, ожидания нет); затем ожидание 900 с: пробуждение
        # тик не запускает, истечение — запускает; третье ожидание прерывает тест.
        assert ticks == ["news_chats", "news_chats"]
        assert waits == [900.0, 900.0, 900.0]

    run(scenario())


def test_tick_logs_summary_of_work_done(monkeypatch, caplog):
    async def scenario():
        async def fake_tick(deps, settings):
            return scheduler.TickResult(analyzed=2, digests=1, urgent=3, disabled_chats=[5])

        monkeypatch.setattr(scheduler, "run_tick", fake_tick)
        sched = scheduler.NewsScheduler(fk_deps(None))
        with caplog.at_level(logging.INFO):
            await sched.tick()
        interval = scheduler.Settings().collect_interval_minutes
        assert sched.next_due == START + timedelta(minutes=interval)

    run(scenario())
    assert "разобрано 2, дайджестов 1, срочных 3, выключено чатов 1" in caplog.text


def test_tick_with_retry_needed_pauses_with_growing_delay_not_beyond_interval(monkeypatch):
    async def scenario():
        async def fake_tick(deps, settings):
            return scheduler.TickResult(retry_needed=True)

        monkeypatch.setattr(scheduler, "run_tick", fake_tick)
        sched = scheduler.NewsScheduler(
            fk_deps(None), scheduler.Settings(collect_interval_minutes=15)
        )
        await sched.tick()
        assert sched.next_due == START + timedelta(seconds=60)
        await sched.tick()
        assert sched.next_due == START + timedelta(seconds=120)
        short = scheduler.NewsScheduler(
            fk_deps(None), scheduler.Settings(collect_interval_minutes=1)
        )
        for _ in range(5):
            await short.tick()
        assert short.next_due == START + timedelta(seconds=60)  # не дольше периода сбора

    run(scenario())


def test_tick_propagates_cancellation(monkeypatch):
    async def scenario():
        async def fake_tick(deps, settings):
            raise asyncio.CancelledError

        monkeypatch.setattr(scheduler, "run_tick", fake_tick)
        with pytest.raises(asyncio.CancelledError):
            await scheduler.NewsScheduler(fk_deps(None)).tick()

    run(scenario())


def test_log_unexpected_end_is_silent_for_cancelled_and_clean_tasks_and_logs_errors(caplog):
    async def scenario():
        async def ok():
            return None

        async def boom():
            raise RuntimeError("упал")

        async def forever():
            await asyncio.sleep(3600)

        done, failed, cancelled = (
            asyncio.ensure_future(ok()),
            asyncio.ensure_future(boom()),
            asyncio.ensure_future(forever()),
        )
        await asyncio.gather(done, failed, return_exceptions=True)
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)
        with caplog.at_level(logging.ERROR):
            scheduler._log_unexpected_end(cancelled)
            scheduler._log_unexpected_end(done)
            assert caplog.text == ""
            scheduler._log_unexpected_end(failed)

    run(scenario())
    assert "неожиданно остановился" in caplog.text


FIXED_NOW = datetime(2026, 10, 18, 15, 30, tzinfo=timezone.utc)


class _FixedNow(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


def fk_candles(first_begin: datetime, closes: list[float]) -> list[dict]:
    """Часовые свечи подряд с московским смещением, как у get_price_history."""
    return [
        {
            "begin": (first_begin + timedelta(hours=i)).astimezone(MSK).isoformat(),
            "open": close,
            "close": close,
        }
        for i, close in enumerate(closes)
    ]


def fk_item(published=None, tickers=({"ticker": "ALFA"},)):
    return {
        "id": "n1",
        "published_at": (published or FIXED_NOW - timedelta(hours=5)).isoformat(),
        "tickers": list(tickers),
    }


def test_build_price_line_describes_change_since_publication_and_requests_history(monkeypatch):
    monkeypatch.setattr(scheduler, "datetime", _FixedNow)

    async def scenario():
        calls = []
        first = datetime(2026, 10, 18, 7, 0, tzinfo=timezone.utc)

        async def call_watch(name, args=None):
            calls.append((name, args))
            return {"candles": fk_candles(first, [100, 100, 100, 100, 110, 110, 110, 110])}

        line = await scheduler.build_price_line(call_watch)(fk_item())
        assert "изменилась на +10%" in line
        assert "совпадение по времени" in line
        assert "не доходность" in line
        assert calls == [
            (
                "get_price_history",
                {"secid": "ALFA", "date_from": "2026-10-14", "interval": "hour"},
            )
        ]

    run(scenario())


def test_build_price_line_without_published_moment_or_tickers_makes_no_request():
    async def scenario():
        calls = []

        async def call_watch(name, args=None):
            calls.append(name)
            return {"candles": []}

        price_line = scheduler.build_price_line(call_watch)
        assert await price_line({"published_at": None, "tickers": [{"ticker": "ALFA"}]}) is None
        assert await price_line({"published_at": "мусор", "tickers": [{"ticker": "ALFA"}]}) is None
        assert await price_line(fk_item(tickers=())) is None
        assert await price_line({"published_at": FIXED_NOW.isoformat()}) is None
        assert calls == []

    run(scenario())


def test_build_price_line_caches_history_per_ticker_until_ttl_expires(monkeypatch):
    monkeypatch.setattr(scheduler, "datetime", _FixedNow)

    async def scenario():
        calls = []
        clock = [1000.0]
        first = datetime(2026, 10, 18, 7, 0, tzinfo=timezone.utc)

        async def call_watch(name, args=None):
            calls.append(args["secid"])
            return {"candles": fk_candles(first, [100] * 4 + [120] * 4)}

        price_line = scheduler.build_price_line(call_watch, clock=lambda: clock[0])
        await price_line(fk_item())
        await price_line(fk_item())
        assert calls == ["ALFA"]
        await price_line(fk_item(tickers=({"ticker": "BETA"},)))
        assert calls == ["ALFA", "BETA"]
        clock[0] += scheduler.PRICE_CACHE_SECONDS + 1
        await price_line(fk_item())
        assert calls == ["ALFA", "BETA", "ALFA"]

    run(scenario())


def test_build_price_line_failure_gives_no_line_and_is_cached_too(monkeypatch):
    monkeypatch.setattr(scheduler, "datetime", _FixedNow)

    async def scenario():
        calls = []

        async def call_watch(name, args=None):
            calls.append(name)
            raise TimeoutError("сервер цен")

        price_line = scheduler.build_price_line(call_watch, clock=lambda: 5.0)
        assert await price_line(fk_item()) is None
        assert await price_line(fk_item()) is None
        assert len(calls) == 1

    run(scenario())


def test_build_price_line_with_empty_candles_has_no_line(monkeypatch):
    monkeypatch.setattr(scheduler, "datetime", _FixedNow)

    async def scenario():
        async def call_watch(name, args=None):
            return {"candles": None}

        assert await scheduler.build_price_line(call_watch)(fk_item()) is None

    run(scenario())


def test_stop_scheduler_without_task_is_a_noop_and_clears_scheduler_key():
    async def scenario():
        app = SimpleNamespace(bot_data={scheduler.SCHEDULER_KEY: object(), "другое": 1})
        await scheduler.stop_scheduler(app)
        assert app.bot_data == {"другое": 1}

    run(scenario())


def test_stop_scheduler_cancels_running_task_and_removes_both_keys():
    async def scenario():
        async def forever():
            await asyncio.sleep(3600)

        task = asyncio.ensure_future(forever())
        await asyncio.sleep(0)
        app = SimpleNamespace(
            bot_data={scheduler.SCHEDULER_KEY: object(), scheduler.TASK_KEY: task}
        )
        await scheduler.stop_scheduler(app)
        assert task.cancelled()
        assert app.bot_data == {}

    run(scenario())
