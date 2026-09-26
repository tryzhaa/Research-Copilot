from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import app
from copilot import demo

PAPER = {"title": "Deep Sets", "year": 2017}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_rate_limiter_slides_its_window() -> None:
    clock = FakeClock()
    limiter = demo.RateLimiter(2, window=60, clock=clock)
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")  # visitors are counted separately
    assert limiter.retry_after("a") == 60
    clock.now = 59.5
    assert not limiter.allow("a")
    clock.now = 60
    assert limiter.allow("a")  # the first event has left the window


@pytest.fixture
def demo_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DEMO_MODE", "1")
    demo.reset()
    yield TestClient(app)
    demo.reset()


@pytest.mark.parametrize("path, body", [
    ("/api/rate", {"paper": PAPER, "rating": 1}),
    ("/api/save", {"paper": PAPER, "saved": True}),
    ("/api/folder", {"paper": PAPER, "folder": "x", "add": True}),
    ("/api/library/remove", {"key": "title:deepsets"}),
])
def test_demo_refuses_every_write(demo_mode: TestClient, path: str, body: dict) -> None:
    r = demo_mode.post(path, json=body)
    assert r.status_code == 403
    assert "read-only demo" in r.json()["detail"]


def test_demo_limits_searches_per_visitor(demo_mode: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_SEARCHES_PER_HOUR", "0")
    demo.reset()
    r = demo_mode.post("/api/search", json={"query": "diffusion", "fields": ["ml"]},
                       headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"})
    assert r.status_code == 429
    assert "searches an hour" in r.json()["detail"]


def test_daily_cap_is_shared_by_all_visitors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "2")
    demo.reset()
    assert demo.spend("search", "alice") is None
    assert demo.spend("summary", "bob") is None
    refusal = demo.spend("search", "carol")
    assert refusal and "today's free model quota" in refusal
    demo.reset()


def test_prefs_tell_the_page_it_is_a_demo(demo_mode: TestClient) -> None:
    assert demo_mode.get("/api/prefs").json()["demo"]["searches_per_hour"] == 5


def test_writes_work_outside_demo_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    from copilot import library
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(library, "PATH", tmp_path / "library.json")  # type: ignore[operator]
    client = TestClient(app)
    assert client.get("/api/prefs").json()["demo"] is None
    assert client.post("/api/rate", json={"paper": PAPER, "rating": 1}).status_code == 200


RANKED = dict(title="Deep Sets", score=8.0, recruiter=9.0, recruiter_reason="impressive", needs_gpu=False,
              compute_note="cpu fine", similarity=0.87, preference=0.7, code_url="https://github.com/x", datasets=["QM9"])


def test_demo_never_sends_the_owners_private_signals(demo_mode: TestClient) -> None:
    from app import serialize
    from copilot.models import Paper
    d = serialize(Paper(**RANKED))  # type: ignore[arg-type]
    assert (d["needs_gpu"], d["compute_note"], d["recruiter"], d["recruiter_reason"], d["similarity"], d["preference"]) \
        == (None, "", None, "", None, None)
    assert d["score"] == 8.0 and d["code_url"] and d["datasets"] == ["QM9"]  # what visitors do see


def test_owner_sees_every_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import serialize
    from copilot.models import Paper
    monkeypatch.delenv("DEMO_MODE", raising=False)
    d = serialize(Paper(**RANKED))  # type: ignore[arg-type]
    assert (d["recruiter"], d["needs_gpu"], d["similarity"], d["recruiter_reason"]) == (9.0, False, 0.87, "impressive")


def test_demo_similar_panel_has_no_similarity_numbers(demo_mode: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from copilot import graph
    monkeypatch.setattr(graph, "similar", lambda paper, key: [
        {"key": "k1", "title": "Near", "url": "u", "year": 2021, "similarity": 0.93, "score": 0.2, "direct": True, "via": None}])
    [item] = demo_mode.post("/api/similar", json={"paper": {"title": "Deep Sets"}}).json()["similar"]
    assert item["similarity"] is None and "score" not in item and item["title"] == "Near"
