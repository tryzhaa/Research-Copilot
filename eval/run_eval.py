"""Does each stage of the ranker actually put the papers you like first? Numbers, per strategy.

    python -m eval.dataset            # refresh eval/dataset.json from your searches + ratings
    python -m eval.run_eval           # compare strategies; appends to eval/results.csv

Each saved search's rated candidates are re-ordered by every strategy below and scored with
P@5, P@10, NDCG@10 and MRR (eval/metrics.py), then averaged over searches (± standard error).
Nothing here calls a model or the network: LLM scores are the ones recorded at search time.

The preference model is scored out-of-fold: for each search, it's retrained without that
search's labeled papers, so it's never graded on a label it was trained on.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import numpy as np

from copilot.models import Paper
from copilot.prefs import load_prefs
from copilot.search import prioritize
from eval import dataset as ds
from eval.metrics import evaluate

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "results.csv"
PAPER_FIELDS = {f.name for f in fields(Paper)}
METRICS = ("p@5", "p@10", "ndcg@10", "mrr")
Cands = list[dict]


def to_paper(c: dict) -> Paper:
    return Paper(**{k: v for k, v in c.items() if k in PAPER_FIELDS})


def by(key: Callable[[dict], float]) -> Callable[[Cands], Cands]:
    """Order by key, highest first; ties keep source order."""
    return lambda cs: sorted(cs, key=lambda c: (-key(c), c["pos"]))


def _num(v: float | None, missing: float = float("-inf")) -> float:
    return missing if v is None else v


def pipeline(priorities: dict, rest_by: str) -> Callable[[Cands], Cands]:
    """What the app shows: the LLM-scored shortlist ordered by prioritize(), then everything the
    LLM never read. Those come after, by similarity (current) or source order (before retrieval)."""
    def order(cs: Cands) -> Cands:
        scored = [c for c in cs if c["shortlisted"] and c["score"] is not None]
        papers = [to_paper(c) for c in scored]
        back = {id(p): c for p, c in zip(papers, scored)}
        top = [back[id(p)] for p in prioritize(papers, priorities)]
        taken = {id(c) for c in top}
        rest = [c for c in cs if id(c) not in taken]
        rest = by(lambda c: _num(c["similarity"]))(rest) if rest_by == "similarity" else sorted(rest, key=lambda c: c["pos"])
        return top + rest
    return order


def strategies(prefs: dict, with_preference: bool) -> dict[str, Callable[[Cands], Cands]]:
    pr = prefs.get("priorities", {})
    w = pr.get("weights", {})
    no_pref = pr | {"weights": w | {"preference": 0.0}}
    original = pr | {"min_relevance": 0, "weights": {"relevance": 0.5, "recruiter": 0.5, "similarity": 0.0, "preference": 0.0}}
    out: dict[str, Callable[[Cands], Cands]] = {
        "random": lambda cs: cs,  # scored as an expectation over shuffles, see run()
        "source order (code first)": lambda cs: sorted(cs, key=lambda c: (not c["code_url"], c["pos"])),
        "recency": by(lambda c: _num(c["year"])),
        "citations": by(lambda c: c["citations"] or 0),
        "similarity only": by(lambda c: _num(c["similarity"])),
        "LLM relevance only": by(lambda c: _num(c["score"])),
        "original heuristic (tiers + 50/50 LLM)": pipeline(original, rest_by="source"),
        "current pipeline (similarity + LLM)": pipeline(no_pref, rest_by="similarity"),
    }
    if with_preference:
        pref_w = w.get("preference") or 0.15
        out["preference model only"] = by(lambda c: _num(c["preference"]))
        out[f"current pipeline + preference ({pref_w:g})"] = pipeline(pr | {"weights": w | {"preference": pref_w}},
                                                                     rest_by="similarity")
    return out


def out_of_fold_preferences(data: dict) -> bool:
    """Set c['preference'] on every candidate from a model trained without that search's labels.
    Returns False when there aren't enough labels to train."""
    from copilot.embed_cache import get_embeddings
    from copilot.embeddings import paper_text
    from copilot.preference_model import can_train, train_preference_model

    labeled = data["labeled"]
    if not labeled:
        return False
    vecs = get_embeddings([p["key"] for p in labeled], [paper_text(p["title"], p["abstract"]) for p in labeled])
    labels = np.array([p["label"] for p in labeled])
    trained_any = False
    for s in data["searches"]:
        held_out = {c["key"] for c in s["candidates"]}
        train = np.array([i for i, p in enumerate(labeled) if p["key"] not in held_out], dtype=int)
        if not can_train(labels[train]):
            for c in s["candidates"]:
                c["preference"] = None
            continue
        clf = train_preference_model(vecs[train], labels[train])
        cand_vecs = get_embeddings([c["key"] for c in s["candidates"]],
                                   [paper_text(c["title"], c["abstract"]) for c in s["candidates"]])
        for c, prob in zip(s["candidates"], clf.predict_proba(cand_vecs)[:, 1]):
            c["preference"] = float(prob)
        trained_any = True
    return trained_any


def run(data: dict, prefs: dict, seed: int = 0, shuffles: int = 200) -> dict[str, dict[str, float]]:
    with_pref = out_of_fold_preferences(data)
    rng = np.random.default_rng(seed)
    per: dict[str, list[dict[str, float]]] = {name: [] for name in strategies(prefs, with_pref)}
    for s in data["searches"]:
        cs = s["candidates"]
        for name, order in strategies(prefs, with_pref).items():
            if name == "random":
                runs = [evaluate([cs[i]["label"] for i in rng.permutation(len(cs))]) for _ in range(shuffles)]
                per[name].append({m: float(np.mean([r[m] for r in runs])) for m in METRICS})
            elif "preference" in name and any(c["preference"] is None for c in cs):
                continue  # not enough other labels to train a model for this search
            else:
                per[name].append(evaluate([c["label"] for c in order(cs)]))

    table = {}
    for name, rows in per.items():
        if not rows:
            continue
        n = len(rows)
        table[name] = {"searches": float(n)}
        for m in METRICS:
            vals = np.array([r[m] for r in rows])
            table[name][m] = float(vals.mean())
            table[name][m + "_se"] = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return table


def markdown(table: dict[str, dict[str, float]]) -> str:
    lines = ["| Strategy | P@5 | P@10 | NDCG@10 | MRR | searches |", "|---|---|---|---|---|---|"]
    for name, r in table.items():
        cells = [f"{r[m]:.3f} ± {r[m + '_se']:.3f}" for m in METRICS]
        lines.append(f"| {name} | {' | '.join(cells)} | {int(r['searches'])} |")
    return "\n".join(lines)


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
        return sha + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_results(table: dict[str, dict[str, float]], stats: dict, path: Path = RESULTS_CSV) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "commit", "labels", "searches", "strategy", *METRICS])
        stamp, commit = time.strftime("%Y-%m-%dT%H:%M:%S"), git_commit()
        for name, r in table.items():
            w.writerow([stamp, commit, stats["labels"], int(r["searches"]), name, *(f"{r[m]:.4f}" for m in METRICS)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ds.PATH)
    ap.add_argument("--build", action="store_true", help="rebuild the dataset from your local searches and ratings first")
    ap.add_argument("--output", type=Path, help="also write the results as JSON here")
    ap.add_argument("--no-log", action="store_true", help="don't append to eval/results.csv")
    args = ap.parse_args(argv)

    if args.build:
        ds.main()
    empty: dict = {"searches": [], "labeled": [], "stats": {"labels": 0}}
    data: dict = json.loads(args.dataset.read_text()) if args.dataset.exists() else empty
    stats: dict = data["stats"]
    if not data["searches"]:
        msg = (f"No usable labeled searches in {args.dataset} ({stats.get('labels', 0)} rated papers). "
               "Search in the app, rate results 👍/👎 (a search needs at least one like and one other rating), "
               "then run `python -m eval.run_eval --build`.")
        print(msg)
        if args.output:
            args.output.write_text(json.dumps({"status": "no_labels", "message": msg, "stats": stats}, indent=1))
        return 0

    table = run(data, load_prefs())
    print(f"{stats['labels']} rated papers, {stats['searches_used']} searches "
          f"({stats.get('leaked_labels_dropped', 0)} labels dropped as seen by the LLM)\n")
    print(markdown(table))
    if not args.no_log:
        log_results(table, stats)
    if args.output:
        args.output.write_text(json.dumps({"status": "ok", "stats": stats, "results": table, "commit": git_commit()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
