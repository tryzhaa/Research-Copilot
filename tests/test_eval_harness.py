"""End to end on synthetic searches with a known answer: likes have high similarity, the LLM
got them backwards, and paper titles carry the like/dislike signal the preference model can learn."""
import numpy as np
import pytest

from copilot import embed_cache
from eval import dataset, run_eval


def _entry(key: str, rating: int) -> dict:
    title = f"{'good' if rating > 0 else 'bad'} paper {key}"
    return {"key": key, "rating": rating, "paper": {"title": title, "abstract": ""}}


def _search(i: int, keys: list[str], liked: set[str], feedback: list[str] = []) -> dict:
    cands = []
    for pos, k in enumerate(keys):
        good = k in liked
        cands.append({
            "key": k, "shortlisted": True, "title": f"{'good' if good else 'bad'} paper {k}", "abstract": "",
            "year": 2020, "citations": 0, "source": "arXiv", "code_url": "", "preference": None,
            "similarity": 0.9 - pos * 0.01 if good else 0.5 - pos * 0.01,
            "score": 2.0 if good else 8.0, "recruiter": 2.0 if good else 8.0,  # the LLM is wrong on purpose
            "datasets": [], "needs_gpu": None,
        })
    # put dislikes first in source order, so source order is also wrong
    cands.sort(key=lambda c: c["key"] in liked)
    return {"id": f"s{i}", "time": 0, "query": f"q{i}", "interests": "", "feedback_titles": feedback, "candidates": cands}


@pytest.fixture
def data() -> dict:
    searches, entries = [], []
    for i in range(6):
        keys = [f"{i}-{j}" for j in range(6)]
        liked = set(keys[:2])
        entries += [_entry(k, 1 if k in liked else -1) for k in keys]
        searches.append(_search(i, keys, liked))
    return dataset.build(searches, entries)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def embed(keys: list[str], texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(len(keys))
        out = rng.normal(0, 0.1, (len(texts), 8))
        out[:, 0] += [1.0 if t.startswith("good") else -1.0 for t in texts]
        return out

    monkeypatch.setattr(embed_cache, "get_embeddings", embed)


def test_dataset_counts(data: dict) -> None:
    assert data["stats"]["labels"] == 36 and data["stats"]["likes"] == 12
    assert data["stats"]["searches_used"] == 6


def test_labels_the_llm_saw_are_dropped() -> None:
    keys = ["a", "b", "c"]
    entries = [_entry("a", 1), _entry("b", -1), _entry("c", 1)]
    s = _search(0, keys, {"a", "c"}, feedback=["good paper a"])
    built = dataset.build([s], entries)
    assert [c["key"] for c in built["searches"][0]["candidates"]] == ["b", "c"]
    assert built["stats"]["leaked_labels_dropped"] == 1


def test_searches_without_a_like_are_skipped() -> None:
    built = dataset.build([_search(0, ["a", "b"], set())], [_entry("a", -1), _entry("b", -1)])
    assert built["searches"] == [] and built["stats"]["searches_too_small"] == 1


def test_strategies_rank_as_constructed(data: dict) -> None:
    prefs = {"priorities": {"code_first": True, "min_datasets": 2, "prefer_cpu": True,
                            "weights": {"relevance": 0.35, "recruiter": 0.35, "similarity": 0.3, "preference": 0.0}}}
    table = run_eval.run(data, prefs)

    assert table["similarity only"]["ndcg@10"] == pytest.approx(1.0)
    assert table["similarity only"]["mrr"] == 1.0
    assert table["LLM relevance only"]["mrr"] == pytest.approx(1 / 5)  # 4 dislikes ranked above
    assert table["source order (code first)"]["mrr"] == pytest.approx(1 / 5)
    rnd = table["random"]["ndcg@10"]
    assert table["LLM relevance only"]["ndcg@10"] < rnd < 1.0
    # the preference model learns "good" from the other five searches, out of fold
    for name in ("preference model only (embeddings)", "preference model only (embeddings + signals)"):
        assert table[name]["ndcg@10"] == pytest.approx(1.0)
        assert table[name]["searches"] == 6


def test_cli_without_data_exits_cleanly(tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
    from pathlib import Path
    out = Path(str(tmp_path)) / "r.json"
    assert run_eval.main(["--dataset", str(Path(str(tmp_path)) / "none.json"), "--output", str(out), "--no-log"]) == 0
    assert '"no_labels"' in out.read_text()
