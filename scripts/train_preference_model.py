"""Train your preference model from the papers you've rated, and report how well it generalizes.

    python -m scripts.train_preference_model                 # embeddings + signals (default)
    python -m scripts.train_preference_model --features embedding

Every rated paper gets its embedding plus its pipeline signals (relevance, recruiter score, code,
datasets, citations, GPU) from the first saved search where the LLM hadn't seen its label. Three
feature sets are cross-validated on identical folds (stratified k-fold ROC-AUC; with ~100 labels a
single split is mostly noise), so you can see what the signals add. Then the chosen set is trained
on everything and saved to data/models/. The app picks it up on the next search; set
`weights.preference` in preferences.yaml to blend it in, and check eval.run_eval to see whether it helps.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from copilot import library, snapshots
from copilot.embed_cache import get_embeddings
from copilot.embeddings import paper_text
from copilot.preference_model import (KINDS, MIN_LABELS, MIN_PER_CLASS, MODEL_PATH, can_train, clean_feature_rows,
                                      cross_validate, features, save_model, train_preference_model)
from eval.run_eval import git_commit

LOG = Path(__file__).resolve().parent.parent / "eval" / "preference_cv.csv"
HEADER = ["timestamp", "commit", "labels", "likes", "folds", "auc", "auc_std", "accuracy", "features"]


def _log(rows: list[list]) -> None:
    """Append to eval/preference_cv.csv. Rows written before feature sets existed were embeddings-only."""
    old: list[list[str]] = []
    if LOG.exists():
        with LOG.open(newline="") as f:
            old = list(csv.reader(f))
        if old and old[0] != HEADER:
            old = [HEADER] + [r + ["embedding"] for r in old[1:]]
    with LOG.open("w", newline="") as f:
        csv.writer(f).writerows((old or [HEADER]) + rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=KINDS, default="enriched", help="which feature set to train and save")
    args = ap.parse_args()

    rated = [e for e in library.entries() if e.get("rating")]
    labels = np.array([int(e["rating"] > 0) for e in rated])
    likes = int(labels.sum())
    print(f"{len(rated)} rated papers: {likes} liked, {len(rated) - likes} disliked")
    if not can_train(labels):
        print(f"Need at least {MIN_LABELS} ratings with {MIN_PER_CLASS}+ of each kind. Rate more papers in the app.")
        return 1
    if len(rated) < 50:
        print("Under 50 labels: expect a noisy model. Keep weights.preference low until the eval says otherwise.")

    signals = clean_feature_rows(rated, snapshots.load_all())
    rows = [signals[e["key"]] for e in rated]
    scored = sum(r["score"] is not None for r in rows)
    print(f"{scored}/{len(rated)} have leakage-free LLM scores; the rest train on embedding, code and citations alone")
    vecs = get_embeddings([e["key"] for e in rated],
                          [paper_text(e["paper"]["title"], e["paper"].get("abstract", "")) for e in rated])

    now, commit, log_rows = time.strftime("%Y-%m-%dT%H:%M:%S"), git_commit(), []
    print("\ncross-validated on identical folds (0.5 = chance):")
    for kind in KINDS:
        cv = cross_validate(features(vecs, rows, kind), labels)
        mark = "  <- saving" if kind == args.features else ""
        print(f"  {kind:10s}  ROC-AUC {cv['auc']:.3f} ± {cv['auc_std']:.3f}, accuracy {cv['accuracy']:.3f}"
              f"  ({int(cv['folds'])} folds){mark}")
        log_rows.append([now, commit, len(rated), likes, int(cv["folds"]), f"{cv['auc']:.4f}",
                         f"{cv['auc_std']:.4f}", f"{cv['accuracy']:.4f}", kind])

    save_model(train_preference_model(features(vecs, rows, args.features), labels), args.features)
    print(f"\nSaved {MODEL_PATH} ({args.features})")
    _log(log_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
