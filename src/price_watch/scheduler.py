"""Планировщик выдачи сводок: часы и доставка опроса цен живут в боте.

Сервер mcp-moex «пассивный»: сам время не измеряет и никому ничего не отправляет.
Бот в нужный момент вызывает `watch_run_due` (сервер делает всё, срок чего наступил,
и называет ближайший следующий срок `next_due_at`), отправляет готовые сводки в чаты
и только после успешной отправки подтверждает их `watch_ack`. См. design.md изменения
add-price-watch-commands, решения 1-4.

Модуль разделён на две части по тому же принципу, что agents/market_tools.py
(`run_tool_loop` там не знает ни про openai, ни про MCP):

- `run_tick()` — ОДИН тик: выдача, отправка, остановка недоступных чатов,
  подтверждение. Не знает ни про Telegram, ни про MCP: вызов инструмента `call_tool` и
  отправка `send` внедряются, поэтому всё ветвление проверяется тестами на простых
  фейках без сна и без сети (tests/test_price_watch_scheduler.py).
- `WatchScheduler` — цикл ожидания вокруг тика: когда проснуться, что делать после
  успеха и после сбоя. Часы и ожидание тоже внедряемые.

Гарантия доставки — «не менее одного раза»: порядок «отправить, затем подтвердить»
означает, что сбой между ними даёт повторную отправку той же сводки, а не потерю.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from telegram.error import BadRequest, Forbidden

from config import TELEGRAM_MESSAGE_LIMIT
from mcp_integration.watch_session import call_watch_tool

logger = logging.getLogger(__name__)

# Ключи application.bot_data: планировщик (его берут обработчики команд) и его задача.
SCHEDULER_KEY = "price_watch_scheduler"
TASK_KEY = "price_watch_scheduler_task"

# Не реже раза в час цикл просыпается и вызывает выдачу, даже если срок далеко или
# сроков нет: страховка от прыжков системных часов и рассинхрона, стоит одного запуска
# сервера в час.
RECHECK_SECONDS = 3600.0
# Пауза перед повтором после сбоя: 60 с, дальше удваивается до 900 с.
RETRY_PAUSE_MIN = 60.0
RETRY_PAUSE_MAX = 900.0
# Сколько раз пробуем подтвердить доставку в одном тике (дубль сводки при неудаче
# неприятнее лишнего вызова сервера).
ACK_ATTEMPTS = 2

TOOL_RUN_DUE = "watch_run_due"
TOOL_ACK = "watch_ack"
TOOL_STOP = "watch_stop"

CallTool = Callable[[str, dict | None], Awaitable[dict]]
Send = Callable[[int, str], Awaitable[None]]
Now = Callable[[], datetime]
Wait = Callable[[float], Awaitable[bool]]


@dataclass(frozen=True)
class TickResult:
    """Итог одного тика: что планировщику делать дальше."""

    next_due_at: datetime | None
    repeat_now: bool = False  # у сервера остались сводки сверх предела одного ответа
    retry_needed: bool = False  # часть доставки не удалась и должна быть повторена позже
    delivered: int = 0
    stopped_chats: list[int] = field(default_factory=list)


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Режет текст на части не длиннее limit, по возможности по границам строк."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def is_chat_unreachable(error: BaseException) -> bool:
    """Telegram сообщает, что писать в этот чат больше нельзя (бот заблокирован
    пользователем, чат не найден). Повторять такую отправку бесполезно."""
    if isinstance(error, Forbidden):
        return True
    return isinstance(error, BadRequest) and "chat not found" in str(error).lower()


def retry_pause(failures: int) -> float:
    """Пауза перед повтором после `failures`-й подряд неудачи: 60, 120, 240, ... не более 900 с."""
    if failures < 1:
        return RETRY_PAUSE_MIN
    return min(RETRY_PAUSE_MIN * 2 ** (failures - 1), RETRY_PAUSE_MAX)


def _parse_moment(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Сервер расписания вернул нераспознанный срок %r.", value)
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


async def _send_report(send: Send, chat_id: int, text: str) -> None:
    chunks = [chunk for chunk in split_message(text) if chunk.strip()]
    for chunk in chunks:
        await send(chat_id, chunk)


async def run_tick(call_tool: CallTool, send: Send) -> TickResult:
    """Один тик: выдача наступивших сводок, отправка, остановка недоступных чатов,
    подтверждение доставленных.

    Исключение поднимается только из самого `watch_run_due` (сервер не запустился,
    тайм-аут, ошибка сервера) — реакция на него (журнал и пауза) в WatchScheduler.
    Сбои внутри тика не роняют его: сбой доставки в один чат не мешает остальным.
    """
    data = await call_tool(TOOL_RUN_DUE, None)
    reports = data.get("reports") or []

    delivered_ids: list[int] = []
    unreachable: list[int] = []
    retry_needed = False
    blocked: set[int] = set()  # чаты, где отправка уже не удалась в этом тике

    for report in reports:
        chat_id = report["chat_id"]
        if chat_id in blocked:
            # Порядок сводок чата сохраняется: после неудачи следующие не отправляем.
            continue
        try:
            await _send_report(send, chat_id, report.get("text") or "")
        except Exception as error:  # noqa: BLE001 — любой сбой отправки не должен ронять тик
            blocked.add(chat_id)
            if is_chat_unreachable(error):
                unreachable.append(chat_id)
                logger.warning(
                    "Чат %s недоступен для отправки сводки (%s): опрос будет остановлен.",
                    chat_id,
                    error,
                )
            else:
                retry_needed = True
                logger.warning(
                    "Не удалось отправить сводку в чат %s: %s. Повтор позже.", chat_id, error
                )
            continue
        delivered_ids.append(report["report_id"])

    stopped: list[int] = []
    for chat_id in unreachable:
        try:
            await call_tool(TOOL_STOP, {"chat_id": chat_id})
            stopped.append(chat_id)
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось остановить опрос недоступного чата %s.", chat_id)
            retry_needed = True

    if delivered_ids:
        for attempt in range(1, ACK_ATTEMPTS + 1):
            try:
                await call_tool(TOOL_ACK, {"report_ids": delivered_ids})
                break
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Не удалось подтвердить доставку сводок %s (попытка %d).",
                    delivered_ids,
                    attempt,
                )
                if attempt == ACK_ATTEMPTS:
                    logger.error("Доставка не подтверждена: сводки будут отправлены повторно.")
                    retry_needed = True

    progressed = bool(delivered_ids) or bool(stopped)
    return TickResult(
        next_due_at=_parse_moment(data.get("next_due_at")),
        # Повторять «сразу», только если что-то сдвинулось: иначе при недоступных
        # получателях цикл крутился бы без пауз.
        repeat_now=bool(data.get("has_more")) and progressed,
        retry_needed=retry_needed or (bool(data.get("has_more")) and not progressed),
        delivered=len(delivered_ids),
        stopped_chats=stopped,
    )


def bot_sender(bot) -> Send:
    """Отправка сообщения в чат через Telegram-бота (обычным текстом, без parse_mode)."""

    async def send(chat_id: int, text: str) -> None:
        await bot.send_message(chat_id=chat_id, text=text)

    return send


class WatchScheduler:
    """Цикл ожидания вокруг тика. Запускается задачей (`asyncio.create_task(run())`)
    в post_init приложения и отменяется в post_shutdown.

    Первый тик — сразу при запуске: доставить сводки, накопившиеся за простой бота, и
    узнать ближайший срок. Дальше цикл ждёт `next_due_at` (но не дольше
    RECHECK_SECONDS), а `schedule_at` может перевести срок на более ранний момент —
    так /watch не заставляет ни ждать час, ни делать лишний вызов сервера «вхолостую».
    """

    def __init__(
        self,
        call_tool: CallTool,
        send: Send,
        *,
        now: Now | None = None,
        wait: Wait | None = None,
    ) -> None:
        self._call_tool = call_tool
        self._send = send
        self._now: Now = now or (lambda: datetime.now(timezone.utc))
        self._wait: Wait = wait or self._wait_for_wake
        self._wake = asyncio.Event()
        self._due: datetime | None = None
        self._failures = 0
        self._in_tick = False
        self._hints_during_tick: list[datetime] = []

    @property
    def next_due(self) -> datetime | None:
        """Запланированный момент следующего тика (None — ждём контрольной сверки)."""
        return self._due

    def schedule_at(self, moment: datetime) -> None:
        """Перевести следующий тик на `moment`, если это раньше запланированного.

        Более поздний момент игнорируется: ближайший тик всё равно наступит раньше и
        получит от сервера свежий срок. Вызывается из обработчика /watch.
        """
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        if self._in_tick:
            # Идущий тик уже получил ответ сервера ДО этой команды: его next_due_at
            # может не знать о новом опросе, поэтому подсказка учитывается по его итогам.
            self._hints_during_tick.append(moment)
        if self._due is None or moment < self._due:
            self._due = moment
        self._wake.set()

    async def _wait_for_wake(self, timeout: float) -> bool:
        """Ждёт `timeout` секунд или пробуждения; True, если разбудили раньше."""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout)
        except TimeoutError:
            return False
        return True

    def _seconds_until_due(self) -> float:
        if self._due is None:
            return RECHECK_SECONDS
        remaining = (self._due - self._now()).total_seconds()
        return min(max(remaining, 0.0), RECHECK_SECONDS)

    async def run(self) -> None:
        """Бесконечный цикл; выходит только по отмене задачи."""
        self._due = self._now()  # первый тик сразу при запуске
        while True:
            timeout = self._seconds_until_due()
            if timeout > 0:
                woken = await self._wait(timeout)
                if woken:
                    self._wake.clear()
                    continue  # срок мог измениться — пересчитать ожидание
            self._wake.clear()
            await self._tick()

    async def _tick(self) -> None:
        self._in_tick = True
        self._hints_during_tick = []
        try:
            result = await run_tick(self._call_tool, self._send)
        except Exception:  # noqa: BLE001 — сбой сервера не должен останавливать планировщик
            self._failures += 1
            pause = retry_pause(self._failures)
            self._due = self._now() + timedelta(seconds=pause)
            logger.exception(
                "Плановый вызов сервера расписания не удался (подряд: %d). Повтор через %.0f с.",
                self._failures,
                pause,
            )
            return
        finally:
            self._in_tick = False

        now = self._now()
        if result.delivered or result.stopped_chats:
            logger.info(
                "Выдача сводок: доставлено %d, остановлено недоступных чатов %d.",
                result.delivered,
                len(result.stopped_chats),
            )
        if result.retry_needed:
            # Часть доставки не удалась: повтор с той же растущей паузой, что и при
            # сбое сервера (не чаще раза в минуту). Подсказки schedule_at и срок
            # сервера здесь не важны: ближайший тик всё равно вызовет выдачу заново.
            self._failures += 1
            self._due = now + timedelta(seconds=retry_pause(self._failures))
            return
        self._failures = 0
        if result.repeat_now:
            self._due = now
            return
        candidates = [moment for moment in (result.next_due_at, *self._hints_during_tick) if moment]
        self._due = min(candidates) if candidates else None


def _log_unexpected_end(task: asyncio.Task) -> None:
    """Цикл планировщика не должен завершаться сам: `_tick` перехватывает сбои. Если он
    всё же умер (ошибка в коде), это должно быть видно в журнале, а не пройти молча."""
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error("Планировщик выдачи сводок неожиданно остановился.", exc_info=error)


async def start_scheduler(application) -> None:
    """post_init приложения: запускает планировщик задачей и кладёт его в bot_data."""
    scheduler = WatchScheduler(call_watch_tool, bot_sender(application.bot))
    task = asyncio.create_task(scheduler.run(), name="price-watch-scheduler")
    task.add_done_callback(_log_unexpected_end)
    application.bot_data[SCHEDULER_KEY] = scheduler
    application.bot_data[TASK_KEY] = task
    logger.info("Опрос цен активен: плановая выдача сводок запущена.")


async def stop_scheduler(application) -> None:
    """post_stop приложения: отменяет задачу и дожидается её завершения. Отмена посреди
    тика допустима: выход из `async with` завершает процесс сервера, а недоставленная
    сводка остаётся неподтверждённой и уйдёт при следующем запуске."""
    task = application.bot_data.pop(TASK_KEY, None)
    application.bot_data.pop(SCHEDULER_KEY, None)
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
