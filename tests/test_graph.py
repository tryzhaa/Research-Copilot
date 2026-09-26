"""The similarity graph on hand-built vectors: two tight topic clusters and a bridge paper."""
import numpy as np
import pytest

from copilot import graph
from copilot.embeddings import normalize


def _graph() -> graph.Graph:
    rng = np.random.default_rng(0)
    a, b = np.eye(16)[0], np.eye(16)[1]
    vecs = [a + rng.normal(0, 0.05, 16) for _ in range(6)]           # topic A: 0-5
    vecs += [b + rng.normal(0, 0.05, 16) for _ in range(6)]          # topic B: 6-11
    vecs += [normalize(a + 0.9 * b) + rng.normal(0, 0.02, 16)]       # 12: between the two
    v = normalize(np.array(vecs))
    keys = [f"k{i}" for i in range(len(v))]
    papers = [{"key": k, "title": f"paper {i}", "year": 2020, "url": f"u{i}", "abstract": ""} for i, k in enumerate(keys)]
    return graph.Graph(keys, papers, v, graph.knn_adjacency(v, k=4, min_sim=0.6), {k: i for i, k in enumerate(keys)})


def test_knn_graph_is_symmetric_without_self_loops_and_respects_the_floor() -> None:
    g = _graph()
    assert (g.adj != g.adj.T).nnz == 0
    assert g.adj.diagonal().sum() == 0
    assert g.adj.data.min() >= 0.6
    rows, cols = g.adj.nonzero()
    assert not any((r < 6 and 6 <= c < 12) for r, c in zip(rows, cols))  # no direct A-B links


def test_pagerank_is_a_distribution_concentrated_near_the_seed() -> None:
    g = _graph()
    seed = np.zeros(13)
    seed[0] = 1
    r = graph.personalized_pagerank(g.adj, seed)
    assert r.sum() == pytest.approx(1.0) and (r >= 0).all()
    assert r[:6].sum() > r[6:12].sum()


def test_similar_returns_close_papers_then_graph_reached_ones_with_via() -> None:
    g = _graph()
    out = graph.similar(g.papers[0], "k0", n_direct=3, n_graph=3, g=g)
    assert all(r["key"] != "k0" for r in out)
    close = [r for r in out if r["direct"]]
    assert len(close) == 3 and all(int(r["key"][1:]) < 6 for r in close)
    for r in out:
        assert r["similarity"] >= 0.6  # nothing drifts in through weak links
        if not r["direct"]:
            assert r["via"] and r["via"].startswith("paper ")


def test_similar_works_for_a_paper_outside_the_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    g = _graph()
    monkeypatch.setattr(graph, "embed_texts", lambda texts: g.vecs[[7]])  # "new" paper looks like topic B
    out = graph.similar({"title": "new"}, "not-in-graph", g=g)
    assert out and all(6 <= int(r["key"][1:]) <= 12 for r in out if r["direct"])


def test_neighbourhood_includes_focus_neighbours_and_only_their_edges() -> None:
    g = _graph()
    data = graph.neighbourhood(["k0"], limit=4, g=g)
    keys = {n["key"] for n in data["nodes"]}
    assert "k0" in keys and len(keys) == 4
    assert [n["focus"] for n in data["nodes"] if n["key"] == "k0"] == [True]
    assert all(l["source"] in keys and l["target"] in keys for l in data["links"])


def test_urls_are_rebuilt_for_old_snapshots() -> None:
    assert graph.paper_url("arxiv:2202.04579", {}) == "https://arxiv.org/abs/2202.04579"
    assert graph.paper_url("doi:10.1/x", {}) == "https://doi.org/10.1/x"
    assert graph.paper_url("title:x", {"title": "Neural Sheaf"}).endswith("q=Neural+Sheaf")
    assert graph.paper_url("arxiv:1", {"url": "https://kept"}) == "https://kept"


def test_related_suggests_unrated_neighbours_of_the_focus_papers() -> None:
    g = _graph()
    found = graph.related(["k0", "k1"], exclude={"k2"}, n=3, g=g)
    keys = [p["key"] for p in found]
    assert len(keys) == 3 and not {"k0", "k1", "k2"} & set(keys)   # never the focus or excluded papers
    assert all(k in {"k3", "k4", "k5", "k12"} for k in keys)        # topic A's neighbours, not topic B
    assert [p["sim"] for p in found] == sorted((p["sim"] for p in found), reverse=True)
    assert all(p["via"] in {"paper 0", "paper 1"} and p["title"] for p in found)


def test_related_is_empty_when_nothing_is_linked() -> None:
    g = _graph()
    assert graph.related(["not-in-graph"], exclude=set(), g=g) == []
