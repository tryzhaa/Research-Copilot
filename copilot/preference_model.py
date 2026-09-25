"""Your taste as a model: logistic regression on frozen paper embeddings, trained on your 👍/👎.

This is a linear probe, not a fine-tune. With tens to low hundreds of labels, a 384-weight
linear model on good embeddings is about as much capacity as the data can support; the
embedding model itself (33M parameters) would memorize them. Train with
`python -m scripts.train_preference_model`; eval/run_eval.py measures whether it helps.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "preference_clf.joblib"
MIN_LABELS = 10       # below this, or without both classes, there's nothing to fit
MIN_PER_CLASS = 3

_cache: dict[Path, tuple[float, LogisticRegression]] = {}


def can_train(labels: np.ndarray) -> bool:
    return len(labels) >= MIN_LABELS and min(int(labels.sum()), int(len(labels) - labels.sum())) >= MIN_PER_CLASS


def train_preference_model(embeddings: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    # balanced: people like far more or far fewer papers than they dislike; don't let the majority win by default
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
    clf.fit(embeddings, labels)
    return clf


def cross_validate(embeddings: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Stratified k-fold ROC-AUC and accuracy. k shrinks with the minority class so every fold has both."""
    minority = int(min(labels.sum(), len(labels) - labels.sum()))
    folds = StratifiedKFold(n_splits=max(2, min(5, minority)), shuffle=True, random_state=seed)
    aucs, accs = [], []
    for train, test in folds.split(embeddings, labels):
        clf = train_preference_model(embeddings[train], labels[train])
        probs = clf.predict_proba(embeddings[test])[:, 1]
        aucs.append(roc_auc_score(labels[test], probs))
        accs.append(float(((probs >= 0.5) == labels[test]).mean()))
    return {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "accuracy": float(np.mean(accs)), "folds": float(folds.n_splits)}


def save_model(clf: LogisticRegression, path: Path = MODEL_PATH) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)


def load_model(path: Path = MODEL_PATH) -> LogisticRegression | None:
    """The trained model, reloaded when the file changes. None if it hasn't been trained."""
    import joblib
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if path not in _cache or _cache[path][0] != mtime:
        _cache[path] = (mtime, joblib.load(path))
    return _cache[path][1]


def predict(embeddings: np.ndarray, path: Path = MODEL_PATH) -> np.ndarray | None:
    """P(you like it) for each row, or None without a trained model."""
    clf = load_model(path)
    if clf is None or len(embeddings) == 0:
        return None
    probs: np.ndarray = clf.predict_proba(embeddings)[:, 1]
    return probs
