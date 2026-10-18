"""Планировщик дайджеста новостей: часы, разбор и доставка живут в боте, сервер новостей пассивен.

Устроен по образцу `price_watch/scheduler.py` (design.md изменения add-news-digest, решения 2, 6).
Модуль разделён на две части:

- `run_tick()` — ОДИН тик: тикеры чатов из опроса цен → сбор лент → разбор свежих новостей →
  срочные сообщения → дайджест → отправка → подтверждение. Не знает ни про Telegram, ни про MCP,
  ни про модель: вызовы серверов, разбор, отправка и часы внедряются (`Deps`), поэтому ветвление
  проверяется тестами на простых фейках (tests/test_news_digest_scheduler.py).
- `NewsScheduler` — цикл ожидания вокруг тика: пауза с удвоением при сбое, как у планировщика цен.

Гарантия доставки — «не менее одного раза»: сначала отправка, затем `news_ack` ТОЛЬКО после успеха.
Сбой отправки оставляет новости неподтверждёнными (уйдут в следующую попытку); потеря недопустима,
редкий повтор допустим. Недоступный чат (бот заблокирован, чат не найден) выключает дайджест ЭТОГО
чата. Остановленный опрос цен выключает дайджест. Пользователям о фоновых сбоях не пишем.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from config import (
    NEWS_COLLECT_INTERVAL_MINUTES,
    NEWS_DIGEST_INTERVAL_HOURS,
    NEWS_DIGEST_MAX_ITEMS,
    NEWS_MAX_ANALYZE_PER_TICK,
    NEWS_MAX_MESSAGES_PER_DAY,
    NEWS_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
)
from mcp_integration.watch_session import call_watch_tool
from price_watch.scheduler import (
    bot_sender,
    is_chat_unreachable,
    retry_pause,
)

from . import analyzer, price_context, rules, texts
from .news_session import call_news_tool

logger = logging.getLogger(__name__)

SCHEDULER_KEY = "news_digest_scheduler"
TASK_KEY = "news_digest_scheduler_task"

# Новость, которую не удалось разобрать из-за недоступной модели, ждёт повтора столько часов от
# публикации, затем показывается без разбора («дайджест не откладывается бесконечно»).
ANALYSIS_GRACE_HOURS = 3
# Сколько готовых к отправке новостей читаем за раз (больше дайджеста: отбор по важности).
READY_FETCH_LIMIT = 50

CallTool = Callable[[str, dict | None], Awaitable[dict]]
Send = Callable[[int, str], Awaitable[None]]
Now = Callable[[], datetime]
Wait = Callable[[float], Awaitable[bool]]
Analyze = Callable[[dict], Awaitable[analyzer.AnalysisOutcome]]
PriceLine = Callable[[dict], Awaitable[str | None]]


@dataclass(frozen=True)
class Settings:
    """Настройки оператора, нужные тику (из config.py)."""

    collect_interval_minutes: int = NEWS_COLLECT_INTERVAL_MINUTES
    digest_interval_hours: int = NEWS_DIGEST_INTERVAL_HOURS
    digest_max_items: int = NEWS_DIGEST_MAX_ITEMS
    max_messages_per_day: int = NEWS_MAX_MESSAGES_PER_DAY
    max_analyze_per_tick: int = NEWS_MAX_ANALYZE_PER_TICK


@dataclass
class Deps:
    call_watch: CallTool  # сервер опроса цен (mcp-moex): тикеры чатов
    call_news: CallTool  # сервер новостей
    analyze: Analyze  # разбор одной новости моделью (под слотом очереди)
    send: Send
    now: Now = lambda: datetime.now(timezone.utc)  # noqa: E731
    price_line: PriceLine | None = None  # строка о совпадении по времени; None — не показывать


@dataclass
class TickResult:
    retry_needed: bool = False  # часть работы не удалась: повторить с паузой
    analyzed: int = 0
    digests: int = 0
    urgent: int = 0
    disabled_chats: list[int] = field(default_factory=list)


async def _price_line(deps: Deps, item: dict) -> str | None:
    """Строка о цене; любой сбой — просто без строки."""
    if deps.price_line is None:
        return None
    try:
        return await deps.price_line(item)
    except Exception:  # noqa: BLE001
        logger.warning("Строка о цене для новости %s недоступна.", item.get("id"), exc_info=True)
        return None


async def _send_all(send: Send, chat_id: int, parts: list[str]) -> None:
    for part in parts:
        if part.strip():
            await send(chat_id, part)


async def _disable_chat(deps: Deps, chat_id: int, result: TickResult, reason: str) -> None:
    try:
        await deps.call_news("news_disable", {"chat_id": chat_id})
        result.disabled_chats.append(chat_id)
        logger.info("Дайджест чата %s выключен: %s.", chat_id, reason)
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось выключить дайджест чата %s.", chat_id)
        result.retry_needed = True


async def _analyze_pending(
    deps: Deps, settings: Settings, tickers: list[str], result: TickResult
) -> None:
    pending = await deps.call_news(
        "news_pending", {"tickers": tickers, "limit": settings.max_analyze_per_tick}
    )
    now = deps.now()
    for news in pending.get("items") or []:
        outcome = await deps.analyze(news)
        if outcome.analysis is None and outcome.transient:
            published = rules.parse_moment(news.get("published_at"))
            stale = published is not None and now - published > timedelta(
                hours=ANALYSIS_GRACE_HOURS
            )
            if not stale:
                # Модель недоступна: не долбим её остальными новостями, повтор на следующем тике.
                result.retry_needed = True
                return
        args = {"news_id": news["id"], "available": outcome.analysis is not None}
        if outcome.analysis is not None:
            args.update(outcome.analysis)
        try:
            await deps.call_news("news_save_analysis", args)
            result.analyzed += 1
        except Exception:  # noqa: BLE001
            logger.warning("Не удалось сохранить разбор новости %s.", news.get("id"), exc_info=True)
            result.retry_needed = True


async def _deliver_chat(
    deps: Deps, settings: Settings, chat_id: int, tickers: list[str], result: TickResult
) -> None:
    now = deps.now()
    state = await deps.call_news("news_state", {"chat_id": chat_id})
    if not state.get("enabled"):
        return
    left = rules.messages_left(int(state.get("messages_today") or 0), settings.max_messages_per_day)
    if left <= 0:
        return
    ready = (
        await deps.call_news(
            "news_ready", {"chat_id": chat_id, "tickers": tickers, "limit": READY_FETCH_LIMIT}
        )
    ).get("items") or []
    if not ready:
        return

    enabled_at = rules.parse_moment(state.get("enabled_at")) or now
    due = now >= rules.next_digest_at(
        enabled_at, rules.parse_moment(state.get("last_digest_at")), settings.digest_interval_hours
    )
    urgent = [item for item in ready if rules.is_urgent(item)]
    has_regular = len(urgent) < len(ready)
    # Одно сообщение суточного лимита бережём для дайджеста, пока есть что в него положить.
    reserve = 1 if (has_regular and due and left >= 2) else 0
    sent_ids: set[str] = set()

    try:
        for item in rules.select_urgent(urgent, left - reserve):
            text = texts.fit_telegram(texts.urgent_message(item, await _price_line(deps, item)))
            await deps.send(chat_id, text)
            await deps.call_news(
                "news_ack", {"chat_id": chat_id, "news_ids": [item["id"]], "kind": "urgent"}
            )
            sent_ids.add(item["id"])
            left -= 1
            result.urgent += 1

        digest_items = rules.select_digest(
            [item for item in ready if item["id"] not in sent_ids], settings.digest_max_items
        )
        if due and left > 0 and digest_items:
            price_lines = {}
            for item in digest_items:
                line = await _price_line(deps, item)
                if line:
                    price_lines[item["id"]] = line
            await _send_all(deps.send, chat_id, texts.digest_messages(digest_items, price_lines))
            await deps.call_news(
                "news_ack",
                {
                    "chat_id": chat_id,
                    "news_ids": [item["id"] for item in digest_items],
                    "kind": "digest",
                },
            )
            result.digests += 1
    except Exception as error:  # noqa: BLE001 — сбой одного чата не должен ронять тик
        if is_chat_unreachable(error):
            logger.warning("Чат %s недоступен (%s): дайджест будет выключен.", chat_id, error)
            await _disable_chat(deps, chat_id, result, "чат недоступен для отправки")
        else:
            logger.warning(
                "Не удалось доставить новости в чат %s: %s. Повтор позже.", chat_id, error
            )
            result.retry_needed = True


async def run_tick(deps: Deps, settings: Settings | None = None) -> TickResult:
    """Один тик. Исключение поднимается только из `news_chats` (сервер новостей не запустился,
    тайм-аут) — реакция (журнал и пауза) в NewsScheduler. Сбои внутри тика (опрос цен, сбор,
    разбор, один чат) его не роняют и помечают `retry_needed`."""
    settings = settings or Settings()
    result = TickResult()
    chats = (await deps.call_news("news_chats", None)).get("chat_ids") or []
    if not chats:
        return result

    watched: dict[int, list[str]] = {}
    for chat_id in chats:
        try:
            status = await deps.call_watch("watch_status", {"chat_id": chat_id})
        except Exception:  # noqa: BLE001
            logger.warning("Не удалось получить опрос цен чата %s.", chat_id, exc_info=True)
            result.retry_needed = True
            continue
        secids = [str(secid).upper() for secid in status.get("secids") or []]
        if status.get("active") and secids:
            watched[chat_id] = secids
        else:
            await _disable_chat(deps, chat_id, result, "опрос цен остановлен")
    if not watched:
        return result

    union = sorted({ticker for tickers in watched.values() for ticker in tickers})
    try:
        report = await deps.call_news("news_collect", {"tickers": union})
        for feed in report.get("feeds_unavailable") or []:
            logger.warning(
                "Лента новостей недоступна: %s — %s.", feed.get("url"), feed.get("error")
            )
    except Exception:  # noqa: BLE001
        logger.warning("Сбор лент новостей не удался.", exc_info=True)
        result.retry_needed = True

    try:
        await _analyze_pending(deps, settings, union, result)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("Разбор новостей прерван сбоем.", exc_info=True)
        result.retry_needed = True

    for chat_id, tickers in watched.items():
        try:
            await _deliver_chat(deps, settings, chat_id, tickers, result)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning("Сбой обработки чата %s.", chat_id, exc_info=True)
            result.retry_needed = True
    return result


class NewsScheduler:
    """Цикл ожидания вокруг тика. Запускается задачей в post_init, отменяется в post_stop.

    Первый тик — сразу при запуске. Дальше — каждые `collect_interval_minutes`; `request_tick`
    (после `/news`) переводит следующий тик на «сейчас». Сбой — пауза 60 с, удвоение до 900 с
    (не дольше периода сбора), сброс после успешного тика.
    """

    def __init__(
        self,
        deps: Deps,
        settings: Settings | None = None,
        *,
        wait: Wait | None = None,
    ) -> None:
        self._deps = deps
        self._settings = settings or Settings()
        self._wait: Wait = wait or self._wait_for_wake
        self._wake = asyncio.Event()
        self._due: datetime | None = None
        self._failures = 0

    @property
    def next_due(self) -> datetime | None:
        return self._due

    def request_tick(self) -> None:
        """Сделать следующий тик немедленным (включили дайджест)."""
        self._due = self._deps.now()
        self._wake.set()

    async def _wait_for_wake(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._wake.wait(), timeout)
        except TimeoutError:
            return False
        return True

    def _seconds_until_due(self) -> float:
        if self._due is None:
            return self._settings.collect_interval_minutes * 60.0
        return max((self._due - self._deps.now()).total_seconds(), 0.0)

    async def run(self) -> None:
        self._due = self._deps.now()
        while True:
            timeout = self._seconds_until_due()
            if timeout > 0:
                woken = await self._wait(timeout)
                if woken:
                    self._wake.clear()
                    continue
            self._wake.clear()
            await self.tick()

    async def tick(self) -> None:
        interval = timedelta(minutes=self._settings.collect_interval_minutes)
        try:
            result = await run_tick(self._deps, self._settings)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — сбой сервера не должен останавливать планировщик
            self._failures += 1
            pause = min(retry_pause(self._failures), interval.total_seconds())
            self._due = self._deps.now() + timedelta(seconds=pause)
            logger.exception(
                "Плановый вызов сервера новостей не удался (подряд: %d). Повтор через %.0f с.",
                self._failures,
                pause,
            )
            return

        if result.analyzed or result.digests or result.urgent or result.disabled_chats:
            logger.info(
                "Новости: разобрано %d, дайджестов %d, срочных %d, выключено чатов %d.",
                result.analyzed,
                result.digests,
                result.urgent,
                len(result.disabled_chats),
            )
        if result.retry_needed:
            self._failures += 1
            pause = min(retry_pause(self._failures), interval.total_seconds())
            self._due = self._deps.now() + timedelta(seconds=pause)
            return
        self._failures = 0
        self._due = self._deps.now() + interval


def _log_unexpected_end(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Планировщик дайджеста новостей неожиданно остановился.", exc_info=error)


def build_deps(application) -> Deps:
    """Зависимости боевого тика: серверы, разбор под слотом общей очереди, отправка в Telegram."""
    from llm_queue.commands import get_gate
    from providers.news_client import news_backend

    gate = get_gate(application)
    complete = analyzer.make_completer(news_backend, NEWS_TIMEOUT_SECONDS) if news_backend else None

    async def analyze(news: dict) -> analyzer.AnalysisOutcome:
        if complete is None:
            return analyzer.AnalysisOutcome(None, 0, "разбор не настроен")
        return await analyzer.analyze_news(news, gate, complete, SYSTEM_PROMPT)

    return Deps(
        call_watch=call_watch_tool,
        call_news=call_news_tool,
        analyze=analyze,
        send=bot_sender(application.bot),
        price_line=build_price_line(call_watch_tool),
    )


PRICE_CACHE_SECONDS = 600.0


def build_price_line(
    call_watch: CallTool, clock: Callable[[], float] = time.monotonic
) -> PriceLine:
    """Строка о цене по свечам `get_price_history` сервера mcp-moex (публичные данные, только
    чтение). Свечи тикера кэшируются на `PRICE_CACHE_SECONDS`, чтобы не запускать сервер на
    каждую новость; сбой запроса — нет строки."""
    cache: dict[tuple[str, str], tuple[float, list[dict] | None]] = {}

    async def price_line(item: dict) -> str | None:
        published = rules.parse_moment(item.get("published_at"))
        tickers = item.get("tickers") or []
        if published is None or not tickers:
            return None
        ticker = tickers[0]["ticker"]
        first_day = (published - timedelta(days=4)).astimezone(price_context.MSK).date().isoformat()
        key = (ticker, first_day)
        cached = cache.get(key)
        if cached is None or clock() - cached[0] > PRICE_CACHE_SECONDS:
            try:
                history = await call_watch(
                    "get_price_history",
                    {"secid": ticker, "date_from": first_day, "interval": "hour"},
                )
                cache[key] = (clock(), history.get("candles") or [])
            except Exception:  # noqa: BLE001
                cache[key] = (clock(), None)
        change = price_context.price_change_since(
            cache[key][1], published, datetime.now(timezone.utc)
        )
        return price_context.coincidence_line(change)

    return price_line


async def start_scheduler(application) -> None:
    """post_init приложения (после start_gate): запускает планировщик задачей."""
    scheduler = NewsScheduler(build_deps(application))
    task = asyncio.create_task(scheduler.run(), name="news-digest-scheduler")
    task.add_done_callback(_log_unexpected_end)
    application.bot_data[SCHEDULER_KEY] = scheduler
    application.bot_data[TASK_KEY] = task
    logger.info("Дайджест новостей активен: плановый сбор и доставка запущены.")


async def stop_scheduler(application) -> None:
    """post_stop приложения (до stop_gate): отменяет задачу и дожидается её завершения. Отмена
    посреди тика допустима: недоставленные новости остаются неподтверждёнными."""
    task = application.bot_data.pop(TASK_KEY, None)
    application.bot_data.pop(SCHEDULER_KEY, None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
