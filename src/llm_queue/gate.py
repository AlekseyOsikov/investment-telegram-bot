"""Шлюз слотов очереди LLM-задач: приоритетное ожидание, реестр задач по чатам, отмена.

Порядок и оценки — чистые правила `rules.py`; здесь только asyncio-часть. Решения —
design.md изменения add-llm-request-queue:

- хендлеры ставят задачу (`submit`) и сразу возвращаются, PTB остаётся последовательным;
- перед слотом задача берёт замок чата: вопросы одного чата выполняются по порядку и
  никогда одновременно (память агента не используется из двух потоков);
- слот освобождается по ФАКТУ завершения работы: отмена asyncio-задачи не останавливает
  поток `asyncio.to_thread`, поэтому при отмене слот и замок чата отдаются по завершении
  потока, а не по отмене;
- очередь живёт только в памяти: чат, вид, время постановки и текст вопроса на время ожидания.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from . import rules
from .rules import INTERACTIVE, KIND_RESEARCH, DurationStats, Estimate, QueueEntry, QueueItemView

logger = logging.getLogger(__name__)

# Ключ application.bot_data, под которым лежит шлюз.
GATE_KEY = "llm_gate"

Notify = Callable[[str], Awaitable[None]]
Late = Callable[[Any, BaseException | None], Awaitable[None]]


@dataclass
class Job:
    """Задача очереди. Живёт, пока держит слот или ждёт его; в реестре чата — пока не снята."""

    seq: int
    chat_id: int
    kind: str
    priority: int
    text: str
    enqueued_at: float
    registered: bool  # учитывается в реестре чата (/queue, лимит чата, /queue_cancel)
    notify: Notify | None = None
    state: str = "waiting"  # waiting | running
    task: asyncio.Task | None = None
    cancelled: bool = False
    cancel_reason: str | None = None
    slot_held: bool = False
    lock: asyncio.Lock | None = None
    started_at: float | None = None
    waited_seconds: float = 0.0
    thread_future: asyncio.Future | None = None
    expired: bool = field(default=False)

    def entry(self) -> QueueEntry:
        return QueueEntry(
            seq=self.seq,
            chat_id=self.chat_id,
            kind=self.kind,
            priority=self.priority,
            running=self.state == "running",
        )


@dataclass(frozen=True)
class Submission:
    """Итог постановки: принят (`accepted`), лимит чата (`chat_limit`) или занято (`busy`)."""

    status: str
    job: Job | None = None
    position: int = 0
    estimate: Estimate | None = None
    queued: bool = False  # True — слот сразу не достанется, пользователю нужен статус


class JobContext:
    """Что получает работа задачи: флаг отмены и запуск блокирующих вызовов в потоке."""

    def __init__(self, gate: LlmGate, job: Job) -> None:
        self._gate = gate
        self._job = job

    @property
    def cancelled(self) -> bool:
        return self._job.cancelled

    @property
    def waited_seconds(self) -> float:
        """Сколько задача ждала слот (для отчётов исследований)."""
        return self._job.waited_seconds

    def should_cancel(self) -> bool:
        """Кооперативная отмена для потоков (`Agent.ask(should_cancel=...)`)."""
        return self._job.cancelled

    async def run_thread(self, fn: Callable[..., Any], *args: Any, late: Late | None = None) -> Any:
        """Выполняет fn(*args) в потоке под слотом задачи.

        Если задачу отменили, пока поток работает, слот и замок чата держатся до конца
        потока; по его завершении вызывается `late(result, exc)` (например, сообщить, что
        обмен уже записан). Отмена при этом пробрасывается как обычно.
        """
        job = self._job
        fut = asyncio.ensure_future(asyncio.to_thread(fn, *args))
        job.thread_future = fut
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            fut.add_done_callback(partial(self._gate._on_thread_done, job, late))
            raise


class LlmGate:
    """Слоты обращений к основной модели с приоритетами, лимитами и оценкой ожидания."""

    def __init__(
        self,
        concurrency: int,
        max_pending_per_user: int,
        queue_max: int,
        ttl_seconds: float,
        timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._concurrency = concurrency
        self._max_pending = max_pending_per_user
        self._queue_max = queue_max
        self._ttl = ttl_seconds
        self._timeout = timeout_seconds
        self._clock = clock
        self._free = concurrency
        self._seq = itertools.count(1)
        self._jobs: dict[int, Job] = {}  # все, кто ждёт или держит слот
        self._by_chat: dict[int, list[Job]] = {}  # реестр чата в порядке отправки
        self._locks: dict[int, asyncio.Lock] = {}
        self._waiters: list[tuple[Job, asyncio.Future]] = []
        self.stats = DurationStats()
        self.user_jobs_done = 0  # обработанных (не снятых) вопросов пользователей с запуска

    # --- наблюдение -------------------------------------------------------------------

    def chat_jobs(self, chat_id: int) -> list[Job]:
        return list(self._by_chat.get(chat_id, ()))

    def _entries(self) -> list[QueueEntry]:
        return [job.entry() for job in self._jobs.values()]

    def _describe(self, job: Job) -> tuple[int, Estimate | None]:
        """Позиция (0 — выполняется) и оценка ожидания ответа (None — выполняется)."""
        entries = self._entries()
        own = job.entry()
        position = rules.queue_position(entries, own)
        if position == 0:
            return 0, None
        ahead = [e.kind for e in rules.entries_ahead(entries, own)]
        estimate = rules.estimate_wait(
            self.stats, ahead, job.kind, self._concurrency, self._timeout
        )
        return position, estimate

    def view_chat(self, chat_id: int) -> list[QueueItemView]:
        """Список вопросов чата для `/queue`: номера подряд с 1 в порядке отправки."""
        items = []
        for number, job in enumerate(self.chat_jobs(chat_id), start=1):
            position, estimate = self._describe(job)
            items.append(QueueItemView(number, position, estimate, job.text))
        return items

    # --- постановка -------------------------------------------------------------------

    def submit(
        self,
        chat_id: int,
        kind: str,
        text: str,
        run: Callable[[JobContext], Awaitable[None]],
        *,
        priority: int = INTERACTIVE,
        notify: Notify | None = None,
    ) -> Submission:
        """Ставит задачу: проверяет лимиты и запускает asyncio-задачу, которая ждёт слот.

        `run(ctx)` выполняется под слотом; ошибки и ответ пользователю она обрабатывает
        сама. `notify(text)` — отправка служебного сообщения чату (срок ожидания истёк).
        """
        chat_jobs = self._by_chat.get(chat_id, [])
        if rules.chat_limit_reached(len(chat_jobs), self._max_pending):
            position, estimate = self._describe(chat_jobs[0])
            return Submission("chat_limit", job=chat_jobs[0], position=position, estimate=estimate)
        if priority == INTERACTIVE and rules.queue_full(
            self._waiting_interactive(), self._queue_max
        ):
            return Submission("busy")

        job = Job(
            seq=next(self._seq),
            chat_id=chat_id,
            kind=kind,
            priority=priority,
            text=text,
            enqueued_at=self._clock(),
            registered=True,
            notify=notify,
        )
        starts_now = self._free > 0 and not self._waiters and not self._chat_lock(chat_id).locked()
        self._jobs[job.seq] = job
        self._by_chat.setdefault(chat_id, []).append(job)
        position, estimate = self._describe(job)
        job.task = asyncio.create_task(self._run_job(job, run))
        return Submission(
            "accepted", job=job, position=position, estimate=estimate, queued=not starts_now
        )

    def _waiting_interactive(self) -> int:
        return sum(
            1 for j in self._jobs.values() if j.state == "waiting" and j.priority == INTERACTIVE
        )

    def _chat_lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = self._locks[chat_id] = asyncio.Lock()
        return lock

    # --- выполнение -------------------------------------------------------------------

    async def _run_job(self, job: Job, run: Callable[[JobContext], Awaitable[None]]) -> None:
        try:
            try:
                await self._wait_turn(job)
            except asyncio.TimeoutError:
                job.expired = True
                self._drop(job)
                logger.info("Вопрос чата %s снят по сроку ожидания", job.chat_id)
                await self._safe_notify(job, rules.text_expired())
                return
            job.state = "running"
            job.started_at = self._clock()
            job.waited_seconds = job.started_at - job.enqueued_at
            await run(JobContext(self, job))
        finally:
            thread = job.thread_future
            if thread is None or thread.done():
                self._finish(job)
            # иначе слот и замок отдаст _on_thread_done по завершении потока

    async def _wait_turn(self, job: Job) -> None:
        """Замок чата, затем слот; для вопросов пользователей — не дольше TTL."""

        async def acquire() -> None:
            lock = self._chat_lock(job.chat_id)
            await lock.acquire()
            job.lock = lock
            await self._acquire_slot(job)

        if job.priority == INTERACTIVE:
            await asyncio.wait_for(acquire(), timeout=self._ttl)
        else:
            await acquire()

    async def _acquire_slot(self, job: Job) -> None:
        if self._free > 0 and not self._waiters:
            self._free -= 1
            job.slot_held = True
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        waiter = (job, fut)
        self._waiters.append(waiter)
        self._dispatch()  # заодно чистит отменённых ожидающих
        try:
            await fut
        except asyncio.CancelledError:
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            else:
                self._release_slot(job)  # слот успели выдать, а задачу отменили
            raise

    def _dispatch(self) -> None:
        self._waiters = [(j, f) for j, f in self._waiters if not f.done()]
        while self._free > 0 and self._waiters:
            best = min(self._waiters, key=lambda w: (w[0].priority, w[0].seq))
            self._waiters.remove(best)
            job, fut = best
            self._free -= 1
            job.slot_held = True
            fut.set_result(None)

    def _release_slot(self, job: Job) -> None:
        if job.slot_held:
            job.slot_held = False
            self._free += 1
            self._dispatch()

    def _finish(self, job: Job) -> None:
        """Освобождает слот и замок чата, пишет длительность, убирает задачу. Идемпотентно."""
        if job.started_at is not None and job.state == "running":
            if not job.cancelled and job.slot_held:
                self.stats.record(job.kind, self._clock() - job.started_at)
                if job.priority == INTERACTIVE and job.registered:
                    self.user_jobs_done += 1
        self._release_slot(job)
        if job.lock is not None and job.lock.locked():
            job.lock.release()
        job.lock = None
        self._drop(job)

    def _drop(self, job: Job) -> None:
        self._jobs.pop(job.seq, None)
        chat_jobs = self._by_chat.get(job.chat_id)
        if chat_jobs is not None:
            if job in chat_jobs:
                chat_jobs.remove(job)
            if not chat_jobs:
                self._by_chat.pop(job.chat_id, None)
        if job.registered and not any(j.chat_id == job.chat_id for j in self._jobs.values()):
            self._locks.pop(job.chat_id, None)

    def _on_thread_done(self, job: Job, late: Late | None, fut: asyncio.Future) -> None:
        """Поток отменённой задачи завершился: теперь можно отдать слот и замок."""
        result: Any = None
        error: BaseException | None = None
        try:
            result = fut.result()
        except BaseException as exc:  # noqa: BLE001 — исключение потока только передаётся дальше
            error = exc
        if late is not None:
            task = asyncio.ensure_future(self._call_late(late, result, error))
            task.add_done_callback(lambda _t: None)
        self._finish(job)

    @staticmethod
    async def _call_late(late: Late, result: Any, error: BaseException | None) -> None:
        try:
            await late(result, error)
        except Exception:
            logger.exception("Сбой уведомления после отмены вопроса")

    async def _safe_notify(self, job: Job, text: str) -> None:
        if job.notify is None:
            return
        try:
            await job.notify(text)
        except Exception:
            logger.warning("Не удалось отправить служебное сообщение очереди", exc_info=True)

    # --- слот для исследований --------------------------------------------------------

    @asynccontextmanager
    async def slot(
        self, priority: int = rules.RESEARCH, kind: str = KIND_RESEARCH, chat_id: int = 0
    ):
        """Слот на один шаг исследования.

        Использование: `async with gate.slot() as ctx: await ctx.run_thread(fn, ...)`.

        Срока ожидания нет; задача не входит в реестр чата (останавливается командой
        остановки прогона — отменой владеющей задачи). `ctx.waited_seconds` — сколько
        шаг ждал слот.
        """
        job = Job(
            seq=next(self._seq),
            chat_id=chat_id,
            kind=kind,
            priority=priority,
            text="",
            enqueued_at=self._clock(),
            registered=False,
        )
        self._jobs[job.seq] = job
        try:
            await self._acquire_slot(job)
        except BaseException:
            self._drop(job)
            raise
        job.state = "running"
        job.started_at = self._clock()
        job.waited_seconds = job.started_at - job.enqueued_at
        try:
            yield JobContext(self, job)
        finally:
            thread = job.thread_future
            if thread is None or thread.done():
                self._finish(job)

    # --- отмена -----------------------------------------------------------------------

    def cancel_job(self, job: Job, reason: str) -> bool:
        """Снимает вопрос. Возвращает True, если он уже выполнялся (слот был у него)."""
        was_running = job.state == "running"
        job.cancelled = True
        job.cancel_reason = reason
        # Из реестра чата — сразу, чтобы /queue и лимит не показывали снятый вопрос;
        # слот и замок чата отдаст завершение работы (_finish).
        chat_jobs = self._by_chat.get(job.chat_id)
        if chat_jobs is not None and job in chat_jobs:
            chat_jobs.remove(job)
            if not chat_jobs:
                self._by_chat.pop(job.chat_id, None)
        if not was_running:
            self._jobs.pop(job.seq, None)
        if job.task is not None and not job.task.done():
            job.task.cancel()
        return was_running

    def cancel_chat(self, chat_id: int, reason: str) -> list[tuple[Job, bool]]:
        """Снимает все вопросы чата; возвращает пары (вопрос, выполнялся ли он)."""
        return [(job, self.cancel_job(job, reason)) for job in self.chat_jobs(chat_id)]

    async def shutdown(self) -> None:
        """Остановка бота: отмена всех задач, ожидание их завершения."""
        tasks = []
        for job in list(self._jobs.values()):
            if job.registered:
                self.cancel_job(job, "shutdown")
            elif job.task is not None and not job.task.done():
                job.task.cancel()
            if job.task is not None:
                tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
