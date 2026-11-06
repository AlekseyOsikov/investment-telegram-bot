"""Rate limiting: не больше N запросов в минуту с одного IP.

Sliding window в памяти процесса. Для демо/одного инстанса этого достаточно;
в проде счётчик должен жить в Redis, иначе лимит «сбрасывается» при рестарте и
не общий для нескольких воркеров.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = max(int(limit), 1)
        self.window_seconds = float(window_seconds)
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Вернуть ``(разрешено, retry_after_seconds)`` и учесть запрос."""
        moment = time.monotonic() if now is None else now
        hits = self._hits[key]
        cutoff = moment - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.limit:
            retry_after = max(0.0, hits[0] + self.window_seconds - moment)
            return False, retry_after

        hits.append(moment)
        return True, 0.0

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._hits.clear()
        else:
            self._hits.pop(key, None)
