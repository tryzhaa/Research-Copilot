"""Build eval/dataset.json from your saved searches (data/searches/) and ratings (library.json).

    python -m eval.dataset

The output is self-contained (titles, abstracts, every stage's scores), so the eval can run
anywhere, including CI, without your library or any network calls. It holds your ratings,
so committing it is your call; CI's eval job only has data if you do.

Leakage: the LLM ranker is shown your recent liked/disliked titles. A candidate whose label
was in that list during a search is dropped from that search's pool, otherwise the LLM would
be graded on answers it was given.
"""
import json
import re
import time
from pathlib import Path

from copilot import library, snapshots

PATH = Path(__file__).resolve().parent / "dataset.json"


def _norm(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())


def build(searches: list[dict], entries: list[dict]) -> dict:
    labels = {e["key"]: int(e["rating"] > 0) for e in entries if e.get("rating")}
    labeled = [{"key": e["key"], "label": labels[e["key"]], "title": e["paper"]["title"],
                "abstract": e["paper"].get("abstract", "")} for e in entries if e["key"] in labels]

    out, leaked, too_small = [], 0, 0
    for s in searches:
        seen = {_norm(t) for t in s.get("feedback_titles", [])}
        judged = []
        for pos, c in enumerate(s["candidates"]):
            if c["key"] not in labels:
                continue
            if _norm(c["title"]) in seen:
                leaked += 1
                continue
            judged.append(c | {"pos": pos, "label": labels[c["key"]]})
        # A pool needs a like to score (NDCG/MRR are undefined without one) and something to order it against.
        if len(judged) < 2 or not any(c["label"] for c in judged):
            too_small += 1
            continue
        out.append({k: s[k] for k in ("id", "time", "query", "interests")} | {"candidates": judged})

    return {
        "built": time.time(),
        "labeled": labeled,
        "searches": out,
        "stats": {"labels": len(labeled), "likes": sum(labels.values()), "searches_total": len(searches),
                  "searches_used": len(out), "searches_too_small": too_small, "leaked_labels_dropped": leaked},
    }


def main() -> None:
    data = build(snapshots.load_all(), library.entries())
    PATH.write_text(json.dumps(data, indent=1))
    s = data["stats"]
    print(f"Wrote {PATH}: {s['labels']} rated papers ({s['likes']} liked), "
          f"{s['searches_used']}/{s['searches_total']} searches usable, "
          f"{s['leaked_labels_dropped']} labels dropped as seen by the LLM.")


if __name__ == "__main__":
    main()
