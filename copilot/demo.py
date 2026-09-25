"""Public demo mode (DEMO_MODE=1), for the hosted copy.

One process serves every visitor, so the demo is read-only: ratings, saves, folders and
removals would otherwise leak between strangers and steer each other's rankings. Searches and
summaries spend the host's free LLM quota, so each visitor gets a few per hour and the whole
demo gets a daily cap. Limits live in memory and reset on restart, which is fine for a demo.
"""
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from functools import cache


def enabled() -> bool:
    return os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class RateLimiter:
    """At most `limit` events per key in any `window` seconds (sliding window)."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _expire(self, key: str, now: float) -> deque[float]:
        events = self._events[key]
        while events and now - events[0] >= self.window:
            events.popleft()
        return events

    def allow(self, key: str) -> bool:
        with self._lock:
            now = self._clock()
            events = self._expire(key, now)
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True

    def retry_after(self, key: str) -> int:
        """Seconds until `key` may try again (0 if it may now)."""
        with self._lock:
            now = self._clock()
            events = self._expire(key, now)
            if len(events) < self.limit or not events:
                return 0
            return max(1, int(events[0] + self.window - now))


@cache
def limits() -> dict[str, int]:
    return {
        "searches_per_hour": _env_int("DEMO_SEARCHES_PER_HOUR", 5),
        "summaries_per_hour": _env_int("DEMO_SUMMARIES_PER_HOUR", 5),
        "llm_calls_per_day": _env_int("DEMO_DAILY_LIMIT", 150),  # searches + summaries, all visitors
    }


@cache
def _limiters() -> dict[str, RateLimiter]:
    lim = limits()
    return {
        "search": RateLimiter(lim["searches_per_hour"], 3600),
        "summary": RateLimiter(lim["summaries_per_hour"], 3600),
        "daily": RateLimiter(lim["llm_calls_per_day"], 86400),
    }


def reset() -> None:
    """Re-read the env and forget all counts (tests)."""
    limits.cache_clear()
    _limiters.cache_clear()


def spend(kind: str, visitor: str) -> str | None:
    """Count one `kind` ("search" or "summary") for `visitor`. Returns why it's refused, or None."""
    per_visitor, daily = _limiters()[kind], _limiters()["daily"]
    if not per_visitor.allow(visitor):
        minutes = -(-per_visitor.retry_after(visitor) // 60)
        plural = {"search": "searches", "summary": "summaries"}[kind]
        return (f"The demo allows {per_visitor.limit} {plural} an hour per visitor. Try again in "
                f"{minutes} min, or run it yourself: github.com/tryzhaa/Research-Copilot")
    if not daily.allow("all"):
        return ("The demo has used today's free model quota. It resets within a day; the code is at "
                "github.com/tryzhaa/Research-Copilot")
    return None
