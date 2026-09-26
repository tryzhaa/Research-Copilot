from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from copilot import sources, trending
from copilot.errors import SourceFetchError

HIT = {"paper": {
    "id": "2412.20138", "title": "TradingAgents:  Multi-Agents LLM\nFinancial Trading", "summary": "An abstract.",
    "authors": [{"name": "A. Author"}, {"_id": "x"}], "publishedAt": "2024-12-28T00:00:00.000Z",
    "githubRepo": "https://github.com/tauricresearch/tradingagents", "githubStars": 108652, "upvotes": 146,
    "ai_summary": "A multi-agent framework\nfor trading.",
}}


@pytest.fixture(autouse=True)
def fresh_cache() -> Iterator[None]:
    trending.clear()
    yield
    trending.clear()


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def test_trending_records_become_papers(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json=[HIT, {"paper": {"title": "no id, skipped"}}])
    [p] = sources.hf_trending(20)
    assert (p.title, p.upvotes, p.stars, p.tldr) == (
        "TradingAgents: Multi-Agents LLM Financial Trading", 146, 108652, "A multi-agent framework for trading.")
    assert p.authors == ["A. Author"] and p.year == 2024 and p.url == "https://huggingface.co/papers/2412.20138"
    request = httpx_mock.get_requests()[0]
    assert request.url.params["sort"] == "trending"


def test_list_is_cached_for_an_hour(httpx_mock: HTTPXMock) -> None:
    clock = Clock()
    httpx_mock.add_response(json=[HIT], is_reusable=True)
    trending.trending(clock)
    clock.now += trending.TTL - 1
    trending.trending(clock)
    assert len(httpx_mock.get_requests()) == 1
    clock.now += 2
    trending.trending(clock)
    assert len(httpx_mock.get_requests()) == 2


def test_stale_list_is_served_when_hugging_face_fails(httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    clock = Clock()
    httpx_mock.add_response(json=[HIT])
    trending.trending(clock)
    clock.now += 3 * 3600
    httpx_mock.add_response(status_code=503, is_reusable=True)
    papers, fetched_at, note = trending.trending(clock)
    assert [p.upvotes for p in papers] == [146]
    assert fetched_at == 1_000_000.0 and note and "3 h old" in note


def test_no_list_at_all_is_a_source_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("offline"))
    with pytest.raises(SourceFetchError, match="Hugging Face"):
        trending.trending()


def test_api_serves_trending_papers(httpx_mock: HTTPXMock) -> None:
    from app import app
    httpx_mock.add_response(json=[HIT])
    body = TestClient(app).get("/api/trending").json()
    [p] = body["papers"]
    assert p["upvotes"] == 146 and p["tldr"] and p["key"] == "arxiv:2412.20138" and body["note"] is None
