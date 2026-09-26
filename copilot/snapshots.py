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

# What the ranking strategies and the similarity graph read, plus what a paper's row shows, so a
# paper clicked on the map can be rebuilt in full. Searches saved before the second line lack it.
KEEP = ("title", "abstract", "year", "url", "citations", "source", "code_url", "similarity", "preference",
        "score", "recruiter", "datasets", "needs_gpu",
        "authors", "venue", "doi", "arxiv_id", "pdf_url", "code_official", "code_framework", "stars",
        "recruiter_reason", "compute_note")


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


def search_id(path: Path) -> str:
    return path.stem.rsplit("-", 1)[-1]


def load_all(directory: Path = DIR) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def load(search_id: str, directory: Path | None = None) -> dict | None:
    """One saved search by its id."""
    if not re.fullmatch(r"[0-9a-f]{12}", search_id):
        return None
    directory = directory or DIR
    found = sorted(directory.glob(f"*-{search_id}.json"))
    return json.loads(found[-1].read_text()) if found else None


def find_paper(key: str, directory: Path | None = None) -> dict | None:
    """The fullest record of a paper you've come across: your library's copy, else its most
    recent appearance in a saved search."""
    from . import library
    if (entry := library.get(key)) is not None:
        return dict(entry["paper"])
    for path in sorted((directory or DIR).glob("*.json"), reverse=True):
        for c in json.loads(path.read_text())["candidates"]:
            if c["key"] == key:
                return {k: v for k, v in c.items() if k not in ("key", "shortlisted")}
    return None
