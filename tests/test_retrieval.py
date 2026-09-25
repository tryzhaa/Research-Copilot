import numpy as np
import pytest

from copilot.embeddings import normalize
from copilot.retrieval import rank_by_similarity
from copilot.search import blended_score, prioritize, shortlist, signals
from tests.conftest import paper


def test_normalize_gives_unit_rows() -> None:
    v = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
    assert np.allclose(v[0], [0.6, 0.8])


def test_cosine_similarity_on_hand_computed_vectors() -> None:
    q = normalize(np.array([1.0, 1.0]))
    docs = normalize(np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [0.0, 1.0]]))
    sims = rank_by_similarity(q, docs)
    assert sims == pytest.approx([2 ** -0.5, 1.0, -1.0, 2 ** -0.5])
    assert list(np.argsort(-sims, kind="stable")) == [1, 0, 3, 2]


def test_shortlist_takes_most_similar_with_code_first() -> None:
    ps = [paper("a", similarity=0.9), paper("b", similarity=0.2, code_url="x"),
          paper("c", similarity=0.5), paper("d", similarity=0.7)]
    assert [p.title for p in shortlist(ps, 3, {"code_first": True})] == ["b", "a", "d"]
    assert [p.title for p in shortlist(ps, 3, {"code_first": False})] == ["a", "d", "c"]


def test_shortlist_without_similarity_keeps_source_order() -> None:
    ps = [paper("a"), paper("b"), paper("c")]
    assert [p.title for p in shortlist(ps, 2, {"code_first": False})] == ["a", "b"]


def test_similarity_is_scaled_within_the_search() -> None:
    s = signals(paper("x", similarity=0.6), (0.4, 0.8))
    assert s["similarity"] == pytest.approx(5.0)


def test_blend_uses_only_the_signals_present() -> None:
    w = {"weights": {"relevance": 0.5, "recruiter": 0.25, "similarity": 0.25, "preference": 0}}
    full = paper("x", score=8.0, recruiter=4.0, similarity=1.0)
    assert blended_score(full, w, (0.0, 1.0)) == pytest.approx(0.5 * 8 + 0.25 * 4 + 0.25 * 10)
    # LLM ranker failed: only similarity is left, so it decides alone
    assert blended_score(paper("y", similarity=0.5), w, (0.0, 1.0)) == pytest.approx(5.0)
    assert blended_score(paper("z"), w) == 0.0


def test_fallback_order_is_by_similarity_when_ranker_is_missing() -> None:
    ps = [paper("low", similarity=0.1), paper("high", similarity=0.9), paper("mid", similarity=0.5)]
    order = prioritize(ps, {"code_first": False, "prefer_cpu": False})
    assert [p.title for p in order] == ["high", "mid", "low"]
