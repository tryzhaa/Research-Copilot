"""Several people using the app at once: shared resources under concurrent requests."""
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from pytest_httpx import HTTPXMock

from copilot import embed_cache, embeddings, library, llm, sources
from copilot.ratelimit import BudgetTimeout, TokenBudget
from tests.conftest import paper

ROOT = Path(__file__).resolve().parent.parent


def run_all(*fns: object) -> None:
    threads = [threading.Thread(target=f) for f in fns]  # type: ignore[arg-type]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)


# ---------- embeddings ----------

def test_cached_lookups_dont_wait_for_someone_elses_embedding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "e.sqlite"
    monkeypatch.setattr(embed_cache, "embed_texts", lambda texts: np.ones((len(texts), 384), dtype=np.float32))
    embed_cache.get_embeddings(["cached"], ["already embedded"], db)

    def slow(texts: list[str]) -> np.ndarray:
        time.sleep(1.0)  # one visitor's search embedding a hundred new papers
        return np.ones((len(texts), 384), dtype=np.float32)
    monkeypatch.setattr(embed_cache, "embed_texts", slow)
    took: dict[str, float] = {}

    def cached_lookup() -> None:
        time.sleep(0.1)  # starts while the slow one is computing
        t = time.monotonic()
        embed_cache.get_embeddings(["cached"], ["already embedded"], db)
        took["cached"] = time.monotonic() - t
    run_all(lambda: embed_cache.get_embeddings(["new"], ["a new paper"], db), cached_lookup)
    assert took["cached"] < 0.3


def test_embedding_runs_one_computation_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    running, peak, lock = [0], [0], threading.Lock()

    def counting(texts: list[str]) -> np.ndarray:
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        return np.ones((len(texts), 384), dtype=np.float32)
    monkeypatch.setattr(embed_cache, "embed_texts", counting)
    db = tmp_path / "e.sqlite"
    run_all(*[lambda i=i: embed_cache.get_embeddings([f"k{i}"], [f"paper {i}"], db) for i in range(4)])
    assert peak[0] == embed_cache.COMPUTE_SLOTS == 1
    assert embed_cache.get_embeddings([f"k{i}" for i in range(4)], [f"paper {i}" for i in range(4)], db).shape == (4, 384)


def test_the_model_loads_once_when_first_requests_arrive_together(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastembed
    loads = []

    class SlowModel:
        def __init__(self, *a: object, **kw: object) -> None:
            time.sleep(0.2)  # loading takes a while: exactly when a second request would slip in
            loads.append(1)
    monkeypatch.setattr(fastembed, "TextEmbedding", SlowModel)
    monkeypatch.setattr(embeddings, "_model", None)
    run_all(*[embeddings.get_model for _ in range(8)])
    assert len(loads) == 1


# ---------- the LLM's shared token budget ----------

class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def test_budget_queues_a_request_until_the_minute_has_room() -> None:
    clock = Clock()
    budget = TokenBudget(8000, clock=clock, sleep=clock.sleep)
    assert budget.acquire(7000, max_wait=120) == 0        # the first ranking goes straight away
    assert budget.acquire(7000, max_wait=120) == 60       # the second waits for the first to age out
    assert budget.acquire(500, max_wait=120) == 0.0       # a small rewrite fits beside it


def test_budget_gives_up_at_the_callers_deadline() -> None:
    clock = Clock()
    budget = TokenBudget(8000, clock=clock, sleep=clock.sleep)
    budget.acquire(7000, max_wait=10)
    with pytest.raises(BudgetTimeout):
        budget.acquire(7000, max_wait=15)                 # a rewrite's 15 s wouldn't reach 60
    assert clock.now == 0                                  # and it didn't wait before saying so


def test_a_request_bigger_than_the_budget_still_runs_alone() -> None:
    clock = Clock()
    budget = TokenBudget(8000, clock=clock, sleep=clock.sleep)
    assert budget.acquire(20000, max_wait=0) == 0


def test_budget_holds_across_threads() -> None:
    budget = TokenBudget(10, window=0.4)
    stamps: list[float] = []
    lock = threading.Lock()

    def take() -> None:
        budget.acquire(5, max_wait=5)
        with lock:
            stamps.append(time.monotonic())
    run_all(*[take for _ in range(6)])  # 30 tokens against 10 per 0.4 s
    stamps.sort()
    assert len(stamps) == 6
    for i in range(len(stamps) - 2):   # never three (15 tokens) inside one window
        assert stamps[i + 2] - stamps[i] >= 0.39


def test_openai_calls_spend_the_shared_budget_and_fail_fast_when_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    from copilot import ratelimit
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(ratelimit, "_budgets", {("groq", 8000): TokenBudget(8000)})
    ratelimit._budgets[("groq", 8000)].acquire(8000, max_wait=0)  # someone else just used the minute
    prefs = {"provider": "groq", "model": "m", "tokens_per_minute": 8000, "llm_deadline": time.monotonic() + 5}
    with pytest.raises(RuntimeError, match="busy with other searches"):
        llm._openai_chat(prefs, "system", "user", max_tokens=100)


def test_rewrites_arent_stuck_behind_slow_rankings(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    monkeypatch.setattr(llm, "rank", lambda *a: release.wait(5) and [])
    monkeypatch.setattr(llm, "_structured", lambda *a: llm.QueryRewrite(keywords="graph networks", intent="i"))
    monkeypatch.setattr(llm, "_rewrites", {})
    prefs = {"provider": "groq", "model": "m"}
    for _ in range(8):  # eight searches' rankings occupying the ranking pool
        llm._rank_pool.submit(llm.rank, [], "q", prefs, [], [])
    t = time.monotonic()
    out = llm.rewrite_query("gnn", prefs, ["ML"], timeout=2)
    release.set()
    assert out.keywords == "graph networks" and time.monotonic() - t < 1


# ---------- paper sources ----------

def test_identical_simultaneous_source_requests_share_one_fetch(httpx_mock: HTTPXMock) -> None:
    def slow_reply(request: object) -> object:
        import httpx
        time.sleep(0.3)
        return httpx.Response(200, json=[])
    httpx_mock.add_callback(slow_reply)
    results: list[list] = []
    run_all(*[lambda: results.append(sources.search_hf_papers("diffusion music", 5, "ml")) for _ in range(5)])
    assert results == [[]] * 5
    assert len(httpx_mock.get_requests()) == 1


def test_cache_expires(httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch) -> None:
    httpx_mock.add_response(json=[], is_reusable=True)
    sources.search_hf_papers("q", 5, "ml")
    sources.search_hf_papers("q", 5, "ml")
    assert len(httpx_mock.get_requests()) == 1
    monkeypatch.setattr(sources, "CACHE_TTL", -1.0)
    sources.search_hf_papers("q", 5, "ml")
    assert len(httpx_mock.get_requests()) == 2


# ---------- the library ----------

def test_threads_saving_at_once_lose_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "PATH", tmp_path / "library.json")
    run_all(*[lambda i=i: library.upsert(paper(f"Paper {i}", doi=f"10.1/{i}"), saved=True) for i in range(20)])
    assert len(library.entries()) == 20


def test_two_server_processes_saving_at_once_lose_nothing(tmp_path: Path) -> None:
    # uvicorn --workers 2, or a script running beside the server: without the file lock, one
    # process's read-modify-write of library.json overwrites the other's.
    path = tmp_path / "library.json"
    worker = (
        "import sys; from pathlib import Path; from copilot import library; from copilot.models import Paper\n"
        f"library.PATH = Path({str(path)!r})\n"
        "for i in range(25):\n"
        "    library.upsert(Paper(title=f'{sys.argv[1]} {i}', doi=f'10.1/{sys.argv[1]}{i}'), saved=True)\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", worker, name], cwd=ROOT) for name in ("a", "b")]
    assert [p.wait(60) for p in procs] == [0, 0]
    monkey = library.PATH
    try:
        library.PATH = path
        assert len(library.entries()) == 50
    finally:
        library.PATH = monkey


def test_requests_arriving_together_book_slots_and_the_hopeless_one_fails_at_once() -> None:
    now = [0.0]
    budget = TokenBudget(8000, clock=lambda: now[0], sleep=lambda s: None)  # everyone arrives at t=0
    assert budget.acquire(7000, max_wait=110) == 0      # search 1 ranks now
    assert budget.acquire(7000, max_wait=110) == 60     # search 2 books the next minute
    t = time.monotonic()
    with pytest.raises(BudgetTimeout, match="120s away"):
        budget.acquire(7000, max_wait=110)              # search 3 would be at 120: told immediately
    assert time.monotonic() - t < 0.1
    assert budget.acquire(800, max_wait=15) == 0        # a query rewrite still fits beside search 1
