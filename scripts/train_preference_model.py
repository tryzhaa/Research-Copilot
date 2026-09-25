"""Train your preference model from the papers you've rated, and report how well it generalizes.

    python -m scripts.train_preference_model

Embeds every rated paper, cross-validates (stratified k-fold ROC-AUC, since with tens of
labels a single split is mostly noise), then trains on everything and saves the model to
data/models/. The app picks it up on the next search; set `weights.preference` in
preferences.yaml to blend it in, and check eval.run_eval to see whether it helps.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

from copilot import library
from copilot.embed_cache import get_embeddings
from copilot.embeddings import paper_text
from copilot.preference_model import MIN_LABELS, MIN_PER_CLASS, MODEL_PATH, can_train, cross_validate, save_model, train_preference_model
from eval.run_eval import git_commit

LOG = Path(__file__).resolve().parent.parent / "eval" / "preference_cv.csv"


def main() -> int:
    rated = [e for e in library.entries() if e.get("rating")]
    labels = np.array([int(e["rating"] > 0) for e in rated])
    likes = int(labels.sum())
    print(f"{len(rated)} rated papers: {likes} liked, {len(rated) - likes} disliked")
    if not can_train(labels):
        print(f"Need at least {MIN_LABELS} ratings with {MIN_PER_CLASS}+ of each kind. Rate more papers in the app.")
        return 1
    if len(rated) < 50:
        print("Under 50 labels: expect a noisy model. Keep weights.preference low until the eval says otherwise.")

    vecs = get_embeddings([e["key"] for e in rated],
                          [paper_text(e["paper"]["title"], e["paper"].get("abstract", "")) for e in rated])
    cv = cross_validate(vecs, labels)
    print(f"cross-validated ROC-AUC {cv['auc']:.3f} ± {cv['auc_std']:.3f}, accuracy {cv['accuracy']:.3f} "
          f"({int(cv['folds'])} folds; 0.5 = chance)")

    save_model(train_preference_model(vecs, labels))
    print(f"Saved {MODEL_PATH}")

    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "commit", "labels", "likes", "folds", "auc", "auc_std", "accuracy"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), git_commit(), len(rated), likes, int(cv["folds"]),
                    f"{cv['auc']:.4f}", f"{cv['auc_std']:.4f}", f"{cv['accuracy']:.4f}"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
