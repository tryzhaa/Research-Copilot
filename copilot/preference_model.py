"""Your taste as a model: logistic regression on your 👍/👎, over the paper's embedding *and* the
signals the pipeline already computes for it.

Features ("enriched"): [embedding (384) | relevance | recruiter | llm_scored | has_code |
log(datasets+1) | log(citations+1) | needs_gpu]. Some of what you like is semantic (the
embedding), but much of it may run through explicit signals, e.g. "papers with code that run on
a CPU", which a probe on raw embeddings can't see. The LLM signals exist only for papers the LLM
scored, so `llm_scored` marks rows where they're real and zeros elsewhere.

This is a linear model, not a fine-tune: with tens to low hundreds of labels that's about the
capacity the data supports. Features are standardized and the regularization strength is chosen
by an inner cross-validation, since ~390 features on ~150 labels overfit easily.
Train with `python -m scripts.train_preference_model`; eval/run_eval.py measures whether it helps.
"""
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "preference_clf.joblib"
MIN_LABELS = 10       # below this, or without both classes, there's nothing to fit
MIN_PER_CLASS = 3

Kind = Literal["embedding", "structured", "enriched"]
KINDS: tuple[Kind, ...] = ("embedding", "structured", "enriched")
STRUCTURED = ("relevance", "recruiter", "llm_scored", "has_code", "log_datasets", "log_citations", "needs_gpu")
# The paper fields structured_features() reads: what snapshots record and eval/dataset.json keeps.
SIGNAL_FIELDS = ("score", "recruiter", "code_url", "datasets", "citations", "needs_gpu")
LLM_FIELDS = ("score", "recruiter", "datasets", "needs_gpu")

_cache: dict[Path, tuple[float, dict]] = {}


def norm_title(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())


# ---------- features ----------

def structured_features(row: Mapping[str, Any]) -> np.ndarray:
    """The explicit signals for one paper, from a Paper.to_dict() or a snapshot candidate."""
    score, recruiter = row.get("score"), row.get("recruiter")
    return np.array([
        score / 10 if score is not None else 0.0,
        recruiter / 10 if recruiter is not None else 0.0,
        float(score is not None),                    # the LLM read it: its signals above are real
        float(bool(row.get("code_url"))),
        math.log1p(len(row.get("datasets") or [])),
        math.log1p(max(0, row.get("citations") or 0)),
        float(row.get("needs_gpu") is True),
    ])


def features(embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]], kind: Kind = "enriched") -> np.ndarray:
    if kind == "embedding":
        return embeddings
    structured = np.array([structured_features(r) for r in rows]).reshape(len(rows), len(STRUCTURED))
    return structured if kind == "structured" else np.hstack([embeddings, structured])


def clean_feature_rows(entries: list[dict], searches: list[dict]) -> dict[str, dict]:
    """For each rated paper, the signals to train on: from the first saved search where the LLM
    scored it *without* having seen its label (the LLM is shown your recent ratings, so a later
    search's score can carry the answer). Failing that, the first clean appearance, then the
    library copy with its LLM signals dropped (it's overwritten on every save, so may be leaked).
    Relevance is query-dependent; taking the first appearance approximates when you judged it."""
    rated = {e["key"]: e for e in entries if e.get("rating")}
    first_scored: dict[str, dict] = {}
    first_seen: dict[str, dict] = {}
    for s in sorted(searches, key=lambda s: s["time"]):
        seen = {norm_title(t) for t in s.get("feedback_titles", [])}
        for c in s["candidates"]:
            if c["key"] not in rated or norm_title(c["title"]) in seen:
                continue
            first_seen.setdefault(c["key"], c)
            if c.get("score") is not None:
                first_scored.setdefault(c["key"], c)
    rows = {}
    for key, e in rated.items():
        src = first_scored.get(key) or first_seen.get(key)
        if src is None:
            src = {k: v for k, v in e["paper"].items() if k not in LLM_FIELDS}
        rows[key] = {k: src.get(k) for k in SIGNAL_FIELDS}
    return rows


# ---------- model ----------

def can_train(labels: np.ndarray) -> bool:
    return len(labels) >= MIN_LABELS and min(int(labels.sum()), int(len(labels) - labels.sum())) >= MIN_PER_CLASS


def train_preference_model(X: np.ndarray, labels: np.ndarray) -> Pipeline:
    """Standardize, then an L2 logistic regression whose strength an inner CV picks by log-loss.
    Log-loss, not ROC-AUC: AUC only sees the ordering, so on easy data every strength ties and the
    strongest wins, squashing probabilities toward 0.5, and the blend uses the probability itself.
    balanced: people like far more or fewer papers than they dislike; don't let the majority win."""
    minority = int(min(labels.sum(), len(labels) - labels.sum()))
    clf = LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0], cv=max(2, min(3, minority)),
                               scoring="neg_log_loss", class_weight="balanced", max_iter=5000,
                               l1_ratios=(0.0,), use_legacy_attributes=False)
    model: Pipeline = make_pipeline(StandardScaler(), clf)
    model.fit(X, labels)
    return model


def cross_validate(X: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Stratified k-fold ROC-AUC and accuracy. k shrinks with the minority class so every fold has both.
    The same seed gives the same folds, so feature sets can be compared on identical splits."""
    minority = int(min(labels.sum(), len(labels) - labels.sum()))
    folds = StratifiedKFold(n_splits=max(2, min(5, minority)), shuffle=True, random_state=seed)
    aucs, accs = [], []
    for train, test in folds.split(X, labels):
        model = train_preference_model(X[train], labels[train])
        probs = model.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(labels[test], probs))
        accs.append(float(((probs >= 0.5) == labels[test]).mean()))
    return {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "accuracy": float(np.mean(accs)), "folds": float(folds.n_splits)}


def save_model(model: Any, kind: Kind, path: Path = MODEL_PATH) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": kind, "model": model}, path)


def load_model(path: Path = MODEL_PATH) -> dict | None:
    """{"kind", "model"}, reloaded when the file changes. None if it hasn't been trained.
    A model saved before features were enriched is a bare classifier on embeddings."""
    import joblib
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if path not in _cache or _cache[path][0] != mtime:
        saved = joblib.load(path)
        _cache[path] = (mtime, saved if isinstance(saved, dict) else {"kind": "embedding", "model": saved})
    return _cache[path][1]


def predict(embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]], path: Path = MODEL_PATH) -> np.ndarray | None:
    """P(you like it) for each paper, or None without a trained model. `rows` are the papers'
    dicts (for the structured signals), aligned with `embeddings`."""
    saved = load_model(path)
    if saved is None or len(embeddings) == 0:
        return None
    probs: np.ndarray = saved["model"].predict_proba(features(embeddings, rows, saved["kind"]))[:, 1]
    return probs
