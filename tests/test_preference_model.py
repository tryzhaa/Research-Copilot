from pathlib import Path

import numpy as np

from copilot.preference_model import can_train, cross_validate, predict, save_model, train_preference_model


def _separable(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.array([i % 3 == 0 for i in range(n)], dtype=int)  # imbalanced: 1 in 3 liked
    X = rng.normal(0, 1, (n, 16))
    X[:, 0] += np.where(y == 1, 2.5, -2.5)
    return X, y


def test_learns_a_separable_preference() -> None:
    X, y = _separable()
    clf = train_preference_model(X[:40], y[:40])
    assert (clf.predict(X[40:]) == y[40:]).mean() >= 0.9


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
    assert predict(X, path) is None  # untrained
    save_model(train_preference_model(X, y), path)
    probs = predict(X, path)
    assert probs is not None and probs.shape == (len(X),)
    assert probs[y == 1].mean() > probs[y == 0].mean()
