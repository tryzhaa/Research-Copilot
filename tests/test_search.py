import httpx
from pytest_httpx import HTTPXMock

from copilot import sources
from copilot.search import apply_filters, dedupe, priority_tier
from tests.conftest import paper

FILTERS = {"min_year": 2015, "min_citations": 5, "exclude_keywords": ["survey"]}


def test_dedupe_merges_arxiv_preprint_with_published_version() -> None:
    preprint = paper("Neural Sheaf Diffusion", arxiv_id="2202.04579v2", source="arXiv", venue="arXiv",
                     pdf_url="https://arxiv.org/pdf/2202.04579")
    published = paper("Neural sheaf diffusion", doi="10.48550/arXiv.2202.04579", citations=120,
                      venue="NeurIPS", source="OpenAlex")
    other = paper("Something else", arxiv_id="1111.11111")
    out = dedupe([preprint, published, other])
    assert len(out) == 2
    merged = out[0]
    assert merged.citations == 120 and merged.venue == "NeurIPS" and merged.pdf_url


def test_dedupe_matches_on_title_when_ids_differ() -> None:
    a = paper("Attention Is All You Need!", doi="10.1/a")
    b = paper("attention is all you need", arxiv_id="1706.03762", code_url="https://github.com/x/y")
    out = dedupe([a, b])
    assert len(out) == 1 and out[0].code_url == "https://github.com/x/y"


def test_filters() -> None:
    ps = [
        paper("old", year=2010, citations=100),
        paper("A survey of GANs", year=2020, citations=100),
        paper("uncited journal", year=2020, citations=1, source="OpenAlex"),
        paper("uncited preprint", year=2024, citations=0, source="arXiv"),  # new preprints are exempt
        paper("good", year=2021, citations=50),
        paper("", year=2021, citations=50),
    ]
    assert [p.title for p in apply_filters(ps, FILTERS, {})] == ["uncited preprint", "good"]
    assert apply_filters([paper("good", year=2021, citations=50)], FILTERS, {"require_code": True}) == []


def test_priority_tiers_order_code_then_datasets_then_cpu() -> None:
    pr = {"code_first": True, "min_datasets": 2, "prefer_cpu": True}
    assert priority_tier(paper("a", code_url="x"), pr) == 4
    assert priority_tier(paper("b", datasets=["MNIST", "CIFAR-10"], needs_gpu=False), pr) == 3
    assert priority_tier(paper("c", needs_gpu=None), pr) == 0  # unknown GPU need isn't rewarded


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2202.04579v2</id>
    <published>2022-02-09T00:00:00Z</published>
    <title>Neural Sheaf
      Diffusion</title>
    <summary>  We study   sheaves. </summary>
    <author><name>Cristian Bodnar</name></author>
  </entry>
</feed>"""


def test_arxiv_parsing(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(text=ATOM)
    [p] = sources.search_arxiv("sheaf diffusion", ["cs.LG"], 5, "ml")
    assert p.title == "Neural Sheaf Diffusion" and p.abstract == "We study sheaves."
    assert p.arxiv_id == "2202.04579v2" and p.year == 2022 and p.authors == ["Cristian Bodnar"]
    q = httpx_mock.get_requests()[0].url.params["search_query"]
    assert q == "all:sheaf AND all:diffusion AND (cat:cs.LG)"


def test_openalex_rebuilds_abstract_from_inverted_index(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json={"results": [{
        "title": "T", "abstract_inverted_index": {"world": [1], "hello": [0]}, "publication_year": 2021,
        "doi": "https://doi.org/10.1/x", "cited_by_count": 3, "authorships": [],
    }]})
    [p] = sources.search_openalex("q", 17, 5, "ml", 2015)
    assert p.abstract == "hello world" and p.doi == "10.1/x"


def test_source_retries_rate_limits_then_succeeds(httpx_mock: HTTPXMock, monkeypatch: object) -> None:
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)  # type: ignore[attr-defined]
    httpx_mock.add_response(status_code=429, headers={"Retry-After": "1"})
    httpx_mock.add_response(json=[])
    assert sources.search_hf_papers("q", 5, "ml") == []
    assert len(httpx_mock.get_requests()) == 2


def test_source_gives_up_after_retries(httpx_mock: HTTPXMock, monkeypatch: object) -> None:
    import time

    import pytest
    monkeypatch.setattr(time, "sleep", lambda s: None)  # type: ignore[attr-defined]
    httpx_mock.add_response(status_code=503, is_reusable=True)
    with pytest.raises(httpx.HTTPStatusError):
        sources.search_hf_papers("q", 5, "ml")
    assert len(httpx_mock.get_requests()) == 4


def test_openalex_scopes_to_subfields_when_given(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json={"results": []}, is_reusable=True)
    sources.search_openalex("q", 12, 5, "art", 2015, [1213, 1210])
    sources.search_openalex("q", 17, 5, "ml", 2015)
    first, second = (r.url.params["filter"] for r in httpx_mock.get_requests())
    assert first == "primary_topic.subfield.id:1213|1210,from_publication_date:2015-01-01"
    assert second == "primary_topic.field.id:17,from_publication_date:2015-01-01"
    assert sources.search_openalex("q", None, 5, "x", 2015) == []


def test_semantic_scholar_accepts_several_fields(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json={"data": []})
    sources.search_semantic_scholar("q", ["Psychology", "Sociology"], 5, "social", 2015)
    assert httpx_mock.get_requests()[0].url.params["fieldsOfStudy"] == "Psychology,Sociology"


def test_openalex_sends_api_key_when_set(httpx_mock: HTTPXMock, monkeypatch: object) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "k123")  # type: ignore[attr-defined]
    httpx_mock.add_response(json={"results": []})
    sources.search_openalex("q", 17, 5, "ml", 2015)
    assert httpx_mock.get_requests()[0].url.params["api_key"] == "k123"
