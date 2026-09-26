from pathlib import Path

import numpy as np

from copilot.preference_model import (STRUCTURED, can_train, clean_feature_rows, cross_validate, features, predict,
                                      save_model, structured_features, train_preference_model)


def _separable(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.array([i % 3 == 0 for i in range(n)], dtype=int)  # imbalanced: 1 in 3 liked
    X = rng.normal(0, 1, (n, 16))
    X[:, 0] += np.where(y == 1, 2.5, -2.5)
    return X, y


def test_learns_a_separable_preference() -> None:
    X, y = _separable(n=300)  # a 150-paper holdout: on 20, one mistake moves accuracy by 5 points
    model = train_preference_model(X[:150], y[:150])
    assert (model.predict(X[150:]) == y[150:]).mean() >= 0.9


def test_cross_validation_beats_chance_on_signal_and_not_on_noise() -> None:
    X, y = _separable()
    assert cross_validate(X, y)["auc"] > 0.95
    rng = np.random.default_rng(1)
    noise = cross_validate(rng.normal(0, 1, X.shape), y)
    assert 0.25 < noise["auc"] < 0.75


def test_can_train_needs_enough_of_each_class() -> None:
    assert not can_train(np.array([1] * 20))
    assert not can_train(np.array([1] * 8 + [0] * 1))
    assert can_train(np.array([1] * 7 + [0] * 3))


def test_predict_round_trips_through_disk(tmp_path: Path) -> None:
    X, y = _separable()
    path = tmp_path / "m.joblib"
    rows = [{}] * len(X)
    assert predict(X, rows, path) is None  # untrained
    save_model(train_preference_model(X, y), "embedding", path)
    probs = predict(X, rows, path)
    assert probs is not None and probs.shape == (len(X),)
    assert probs[y == 1].mean() > probs[y == 0].mean()



def test_structured_features_encode_signals_and_mark_unscored_papers() -> None:
    scored = {"score": 8.0, "recruiter": 6.0, "code_url": "https://github.com/x", "datasets": ["QM9", "ZINC"],
              "citations": 99, "needs_gpu": False}
    f = dict(zip(STRUCTURED, structured_features(scored)))
    assert f["relevance"] == 0.8 and f["recruiter"] == 0.6 and f["llm_scored"] == 1.0 and f["has_code"] == 1.0
    assert f["log_datasets"] == np.log1p(2) and f["log_citations"] == np.log1p(99) and f["needs_gpu"] == 0.0
    unscored = dict(zip(STRUCTURED, structured_features({"citations": None, "needs_gpu": None})))
    assert unscored == dict.fromkeys(STRUCTURED, 0.0)  # llm_scored = 0 says the LLM zeros aren't real


def test_enriched_features_catch_a_preference_the_embeddings_miss() -> None:
    # You like papers with code that run on a CPU; the embeddings carry no signal at all.
    rng = np.random.default_rng(0)
    n = 160
    rows = [{"score": float(rng.integers(3, 10)), "recruiter": float(rng.integers(3, 10)),
             "code_url": "x" if rng.random() < 0.5 else "", "datasets": [], "citations": int(rng.integers(0, 500)),
             "needs_gpu": bool(rng.random() < 0.5)} for _ in range(n)]
    y = np.array([int(bool(r["code_url"]) and not r["needs_gpu"]) for r in rows])
    emb = rng.normal(0, 1, (n, 32))
    assert cross_validate(features(emb, rows, "embedding"), y)["auc"] < 0.7
    assert cross_validate(features(emb, rows, "enriched"), y)["auc"] > 0.9


def test_signals_come_from_the_first_search_that_hadnt_seen_the_label() -> None:
    entries = [
        {"key": "a", "rating": 1, "paper": {"title": "A", "score": 9.9, "code_url": "lib", "citations": 5}},
        {"key": "b", "rating": -1, "paper": {"title": "B", "score": 1.0, "recruiter": 1.0, "code_url": "", "citations": 7}},
    ]
    cand = lambda key, title, score: {"key": key, "title": title, "score": score, "recruiter": 5.0,
                                      "code_url": "", "datasets": ["QM9"], "citations": 1, "needs_gpu": False}
    searches = [
        {"time": 3, "feedback_titles": [], "candidates": [cand("a", "A", 7.0)]},
        {"time": 1, "feedback_titles": ["A"], "candidates": [cand("a", "A", 9.0)]},  # the LLM saw A's label
        {"time": 2, "feedback_titles": [], "candidates": [cand("a", "A", None)]},     # clean, but not scored
    ]
    rows = clean_feature_rows(entries, searches)
    assert rows["a"]["score"] == 7.0  # first clean *scored* appearance, not the earlier leaked one
    # b never appeared in a search: library copy, with its (possibly leaked) LLM signals dropped
    assert rows["b"] == {"score": None, "recruiter": None, "code_url": "", "datasets": None, "citations": 7,
                         "needs_gpu": None}


def test_a_model_saved_before_enrichment_still_loads(tmp_path: Path) -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    X, y = _separable()
    path = tmp_path / "old.joblib"
    joblib.dump(LogisticRegression().fit(X, y), path)  # the old format: a bare classifier on embeddings
    probs = predict(X, [{}] * len(X), path)
    assert probs is not None and probs[y == 1].mean() > probs[y == 0].mean()
