"""A token-per-minute budget shared by every request to a hosted LLM.

Free tiers cap tokens per minute across the whole API key (Groq free: ~8,000), and one ranking
uses ~7,000. Without coordination, simultaneous searches all fire at once, all but one get a
429, and those searches silently fall back to similarity-only results. With a shared budget
they book slots instead: a request that can be served before its caller's deadline waits for
its slot; one that can't gives up immediately (the search ranks by similarity in seconds, not
after a long wait), and never spends budget after its caller has moved on.
"""
import threading
import time
from collections.abc import Callable


class BudgetTimeout(Exception):
    """The budget won't have room before the caller's deadline."""


class TokenBudget:
    """Reservations against a sliding window: a request books the earliest moment its tokens fit,
    counting everyone already booked, including those booked for later. So a request that
    can't be served before its deadline knows at once and gives up in milliseconds instead of
    waiting it out, and a small request (a query rewrite) can still fit beside a queued big one."""

    def __init__(self, per_minute: int, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.per_minute = per_minute
        self.window = window
        self._clock, self._sleep = clock, sleep
        self._booked: list[tuple[float, int]] = []  # (send time, tokens), in time order
        self._lock = threading.Lock()

    def _load(self, at: float) -> int:
        """Tokens booked in the window ending at `at`."""
        return sum(t for when, t in self._booked if at - self.window < when <= at)

    def _earliest(self, tokens: int, now: float) -> float:
        """The first moment `tokens` fit without pushing any window over budget. Lock held."""
        self._booked = [(w, t) for w, t in self._booked if now - w < self.window]  # forget the expired
        for start in sorted({now} | {w + self.window for w, _ in self._booked}):
            if start < now:
                continue
            # The new tokens count in every window ending in [start, start + window): check
            # the start and each booking inside that span (the load only changes there).
            ends = [start] + [w for w, _ in self._booked if start <= w < start + self.window]
            if all(self._load(end) + tokens <= self.per_minute for end in ends):
                return start
        raise AssertionError("unreachable: after every booking expires, anything up to the budget fits")

    def acquire(self, tokens: int, max_wait: float) -> float:
        """Book `tokens`, then wait until the booked moment. Raises BudgetTimeout at once, without
        booking, if that moment is more than `max_wait` away. Returns the seconds waited. A request
        larger than the whole budget is let through alone rather than refused forever."""
        tokens = min(tokens, self.per_minute)
        with self._lock:
            now = self._clock()
            at = self._earliest(tokens, now)
            if at - now > max_wait:
                raise BudgetTimeout(f"the next free slot is {at - now:.0f}s away")
            self._booked.append((at, tokens))
            self._booked.sort()
        if at > now:
            self._sleep(at - now)
        return at - now


_budgets: dict[tuple[str, int], TokenBudget] = {}
_budgets_lock = threading.Lock()


def budget_for(provider: str, per_minute: int) -> TokenBudget:
    """One shared budget per provider and limit, for the life of the process."""
    with _budgets_lock:
        return _budgets.setdefault((provider, per_minute), TokenBudget(per_minute))
