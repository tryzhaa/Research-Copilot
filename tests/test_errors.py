import httpx
import pytest

from copilot import search
from copilot.errors import RankingTimeoutError, SourceFetchError, classify
from tests.conftest import paper


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://example.org")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(code, request=req))


@pytest.mark.parametrize("exc, kind", [
    (httpx.ReadTimeout("slow"), "timeout"),
    (_status_error(429), "rate_limit"),
    (_status_error(503), "http_error"),
    (httpx.ConnectError("down"), "network"),
    (KeyError("results"), "bad_response"),
])
def test_classify(exc: Exception, kind: str) -> None:
    assert classify(exc) == kind


def test_source_error_serializes_for_the_ui() -> None:
    d = SourceFetchError("OpenAlex", "ml", _status_error(429)).to_dict()
    assert d == {"source": "OpenAlex", "field": "ml", "error_type": "rate_limit",
                 "message": "OpenAlex (ml): rate limited, try again in a minute"}


def test_timeout_is_a_ranking_error() -> None:
    assert RankingTimeoutError("x").to_dict()["error_type"] == "timeout"
    assert RankingTimeoutError("x").to_dict()["source"] == "ranker"


def test_a_failing_source_is_reported_and_the_rest_still_return(monkeypatch: pytest.MonkeyPatch, prefs: dict) -> None:
    def broken(*_: object) -> list:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(search, "search_arxiv", broken)
    monkeypatch.setattr(search, "search_openalex", lambda *_: [paper("Topology of GANs", year=2020)])
    monkeypatch.setattr(search, "search_hf_papers", lambda *_: [])
    monkeypatch.setattr(search.pwc, "lookup", lambda ids: {})

    papers, errors = search.search_all("gans", prefs, ["ml"])

    assert [p.title for p in papers] == ["Topology of GANs"]
    assert [(e.source, e.error_type) for e in errors] == [("arXiv", "network")]
