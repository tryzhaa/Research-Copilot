"""Every search is saved to data/searches/ as an evaluation record: the query, every candidate
with the scores each stage gave it, and which of your ratings the LLM saw at the time.

Your ratings (library.json) later label these candidates. eval/ replays each search's
labeled pool under different ranking strategies and measures which orders your likes first.
"""
import json
import re
import time
import uuid
from pathlib import Path

from .models import Paper

DIR = Path(__file__).resolve().parent.parent / "data" / "searches"

# Only what the ranking strategies read — keeps snapshots small and the eval reproducible.
KEEP = ("title", "abstract", "year", "citations", "source", "code_url", "similarity", "preference",
        "score", "recruiter", "datasets", "needs_gpu")


def save(query: str, prefs: dict, fields: list[str], pool: list[Paper], ranked: list[Paper],
         feedback_titles: list[str], rewrite: dict | None = None, directory: Path = DIR) -> Path:
    """pool: every candidate after filtering, in source order. ranked: the shortlist the LLM scored."""
    scored = {p.key: p for p in ranked}
    record = {
        "id": uuid.uuid4().hex[:12],
        "time": time.time(),
        "query": query,
        "rewrite": rewrite,  # {keywords, intent} the sources and similarity actually used, or None
        "interests": prefs.get("interests", ""),
        "fields": fields,
        "provider": prefs.get("provider", "ollama"),
        "model": prefs.get("model", ""),
        "weights": prefs.get("priorities", {}).get("weights", {}),
        "feedback_titles": feedback_titles,  # the LLM saw these labels: excluded when evaluating this search
        "candidates": [
            {"key": p.key, "shortlisted": p.key in scored,
             **{k: getattr(scored.get(p.key, p), k) for k in KEEP}}
            for p in pool
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"\W+", "-", query.lower()).strip("-")[:40]
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}-{record['id']}.json"
    path.write_text(json.dumps(record))
    return path


def load_all(directory: Path = DIR) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
