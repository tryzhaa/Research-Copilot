"""Trending papers for the home screen, before anything is searched.

Hugging Face's trending list, cached for an hour: the page loads it on every visit, and it
changes slowly. No LLM is involved, so it costs nothing and doesn't count against the demo's
limits. If Hugging Face doesn't answer, the last good list is served with a note.
"""
import threading
import time
from collections.abc import Callable

from .errors import SourceFetchError
from .models import Paper
from .search import attach_code
from .sources import hf_trending

TTL = 3600
LIMIT = 20

_lock = threading.Lock()
_cache: tuple[float, list[Paper]] | None = None


def trending(clock: Callable[[], float] = time.time) -> tuple[list[Paper], float, str | None]:
    """(papers, fetched_at, note). The note says when a stale list is being shown."""
    global _cache
    with _lock:  # one fetch at a time; concurrent visitors wait for it rather than all fetching
        now = clock()
        if _cache and now - _cache[0] < TTL:
            return _cache[1], _cache[0], None
        try:
            papers = hf_trending(LIMIT)
        except Exception as e:
            if _cache:
                hours = max(1, round((now - _cache[0]) / 3600))
                return _cache[1], _cache[0], f"Hugging Face didn't answer; this list is {hours} h old"
            raise SourceFetchError("Hugging Face", "trending", e) from e
        attach_code(papers)  # official repos from the Papers with Code index, where HF has none
        _cache = (now, papers)
        return papers, now, None


def clear() -> None:
    global _cache
    with _lock:
        _cache = None
