"""A paper clicked on the map joins the search results, ranked like them."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app
from copilot import demo, library, llm, retrieval, snapshots
from copilot.models import Paper
from copilot.search import prioritize, rank_value, similarity_range
from tests.conftest import paper

PRIORITIES = {"tier_order": ["code", "datasets", "cpu"], "min_datasets": 1, "min_relevance": 5,
              "weights": {"relevance": 0.3, "recruiter": 0.6, "similarity": 0.3}}


def test_rank_value_orders_like_prioritize() -> None:
    ps = [paper("off-topic, code", score=2.0, recruiter=9.0, code_url="x", similarity=0.5),
          paper("on-topic, code", score=6.0, recruiter=4.0, code_url="y", similarity=0.6),
          paper("on-topic, no code, great", score=9.0, recruiter=9.0, similarity=0.9, datasets=["QM9"]),
          paper("unscored, similar", similarity=0.8)]
    rng = similarity_range(ps)
    by_value = sorted(ps, key=lambda p: rank_value(p, PRIORITIES, rng), reverse=True)
    assert [p.title for p in by_value] == [p.title for p in prioritize(ps, PRIORITIES)]
    assert by_value[0].title == "on-topic, code"  # code leads the tiers


@pytest.fixture
def saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """One saved search, with the clicked paper known from an earlier one."""
    monkeypatch.setattr(snapshots, "DIR", tmp_path / "searches")
    monkeypatch.setattr(library, "PATH", tmp_path / "library.json")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(app_module, "load_prefs", lambda: {
        "priorities": PRIORITIES, "interests": "molecules", "provider": "groq", "model": "m", "show_top": 10})
    earlier = [paper("GemNet", arxiv_id="2106.08903", authors=["J. Gasteiger"], venue="NeurIPS",
                     score=3.0, reason="about something else", recruiter=8.0, datasets=["MD17"], needs_gpu=True)]
    snapshots.save("force fields", {}, ["ml"], earlier, earlier, [], directory=snapshots.DIR)
    now = [paper("Top", score=9.0, recruiter=9.0, code_url="a", similarity=0.9),
           paper("Middle", score=7.0, recruiter=5.0, similarity=0.7),
           paper("Bottom", score=6.0, recruiter=2.0, similarity=0.4)]
    path = snapshots.save("gnn molecules", {"interests": "molecules"}, ["ml"], now, now, [],
                          rewrite={"keywords": "gnn molecules", "intent": "graph networks for molecules"},
                          directory=snapshots.DIR)
    seen: dict = {}

    def fake_similarity(papers: list[Paper], query: str, interests: str) -> None:
        seen["similarity_query"] = query
        papers[0].similarity = 0.8

    def fake_rank(papers: list[Paper], query: str, prefs: dict, liked: list, disliked: list, timeout: float) -> list[Paper]:
        seen["rank_query"] = query
        papers[0].score, papers[0].reason, papers[0].recruiter = 8.0, "fits", 7.0
        papers[0].datasets = ["MD17", "OC20"]
        return papers
    monkeypatch.setattr(retrieval, "score_similarity", fake_similarity)
    monkeypatch.setattr(retrieval, "score_preference", lambda papers: None)
    monkeypatch.setattr(app_module, "attach_code", lambda papers: None)  # not the local Papers with Code index
    monkeypatch.setattr(app_module, "rank_with_timeout", fake_rank)
    return {"id": snapshots.search_id(path), "seen": seen}


def test_a_map_paper_comes_back_in_full_and_ranked_for_this_search(saved: dict) -> None:
    key = Paper(title="GemNet", arxiv_id="2106.08903").key
    r = TestClient(app).post("/api/place", json={"search_id": saved["id"], "key": key})
    assert r.status_code == 200, r.text
    p = r.json()["paper"]
    assert p["authors"] == ["J. Gasteiger"] and p["venue"] == "NeurIPS"   # the fullest saved record
    assert (p["score"], p["reason"], p["datasets"]) == (8.0, "fits", ["MD17", "OC20"])  # read for *this* query
    assert "gnn molecules" in saved["seen"]["rank_query"]
    assert saved["seen"]["similarity_query"] == "graph networks for molecules"
    snap = snapshots.load(saved["id"])
    ranks = {c["title"]: rank_value(Paper.from_dict(c), PRIORITIES, similarity_range(
        [Paper(title=x["title"], similarity=x["similarity"]) for x in snap["candidates"]])) for c in snap["candidates"]}
    assert ranks["Top"] > p["rank"] > ranks["Middle"]   # slots in where the ranking puts it


def test_when_the_model_is_busy_it_keeps_its_stored_signals_but_not_the_old_relevance(saved: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from copilot.errors import RankingError

    def busy(*a: object) -> list[Paper]:
        raise RankingError("busy with other searches")
    monkeypatch.setattr(app_module, "rank_with_timeout", busy)
    key = Paper(title="GemNet", arxiv_id="2106.08903").key
    data = TestClient(app).post("/api/place", json={"search_id": saved["id"], "key": key}).json()
    p = data["paper"]
    assert p["score"] is None and p["reason"] == ""       # its relevance was to "force fields"
    assert p["recruiter"] == 8.0 and p["datasets"] == ["MD17"]
    assert "busy" in data["errors"][0]["message"]


def test_unknown_search_or_paper_is_404(saved: dict) -> None:
    client = TestClient(app)
    assert client.post("/api/place", json={"search_id": "000000000000", "key": "x"}).status_code == 404
    assert client.post("/api/place", json={"search_id": "../../etc", "key": "x"}).status_code == 404
    assert client.post("/api/place", json={"search_id": saved["id"], "key": "title:nothing"}).status_code == 404


def test_search_results_carry_their_rank_and_the_search_id(saved: dict) -> None:
    # (the full /api/search is covered elsewhere; here: saved searches can be found again by id)
    assert snapshots.load(saved["id"])["query"] == "gnn molecules"
    assert snapshots.find_paper(Paper(title="GemNet", arxiv_id="2106.08903").key)["authors"] == ["J. Gasteiger"]


def test_demo_counts_map_adds_on_their_own_allowance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MAP_ADDS_PER_HOUR", "1")
    demo.reset()
    assert demo.spend("place", "v") is None
    assert "papers added from the map an hour" in (demo.spend("place", "v") or "")
    assert demo.spend("search", "v") is None   # searches are counted separately
    demo.reset()
