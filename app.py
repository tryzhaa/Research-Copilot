"""Research Copilot web app. Run: uvicorn app:app --reload  →  http://localhost:8000"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import fields
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from copilot import graph, library, pwc, retrieval, snapshots
from copilot.errors import EmbeddingError, RankingError, RewriteError
from copilot.llm import QueryRewrite, rank_with_timeout, rewrite_query, summarize
from copilot.models import Paper
from copilot.prefs import load_prefs
from copilot.search import prioritize, search_all, shortlist

ROOT = Path(__file__).parent
log = logging.getLogger("uvicorn.error")
PAPER_FIELDS = {f.name for f in fields(Paper)}

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    pwc.build_in_background()  # one-time Papers with Code index; searches work without it meanwhile
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def to_paper(d: dict) -> Paper:
    return Paper(**{k: v for k, v in d.items() if k in PAPER_FIELDS})


def serialize(p: Paper) -> dict:
    entry = library.get(p.key) or {}
    return p.to_dict() | {
        "key": p.key,
        "has_code": p.has_code,
        "bibtex": p.bibtex(),
        "rating": entry.get("rating", 0),
        "saved": entry.get("saved", False),
    }


class SearchIn(BaseModel):
    query: str
    fields: list[str]
    use_s2: bool = False
    code_only: bool = False


class PaperIn(BaseModel):
    paper: dict
    refresh: bool = False


class RateIn(BaseModel):
    paper: dict
    rating: int  # -1, 0, 1


class SaveIn(BaseModel):
    paper: dict
    saved: bool


class KeyIn(BaseModel):
    key: str


class SimilarIn(BaseModel):
    paper: dict
    key: str | None = None  # the map sends graph keys directly


class MapIn(BaseModel):
    keys: list[str]  # papers on screen; the library is always included


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/prefs")
def get_prefs() -> dict:
    prefs = load_prefs()
    return {
        "fields": {k: v["label"] for k, v in prefs["fields"].items()},
        "provider": prefs.get("provider", "ollama"),
        "model": prefs["model"],
        "effort": prefs.get("effort", "medium"),
    }


@app.post("/api/search")
def search(body: SearchIn) -> dict:
    prefs = load_prefs()
    field_keys = [k for k in body.fields if k in prefs["fields"]]
    if not body.query.strip() or not field_keys:
        raise HTTPException(400, "Enter a query and pick at least one field.")
    priorities = prefs.get("priorities", {}) | ({"require_code": True} if body.code_only else {})
    query = body.query.strip()
    errors: list[dict] = []

    # Rewrite: keywords for the sources, an intent sentence for similarity and the ranker.
    rewrite: QueryRewrite | None = None
    if prefs.get("rewrite_query", True):
        try:
            rewrite = rewrite_query(query, prefs, [prefs["fields"][k]["label"] for k in field_keys],
                                    prefs.get("rewrite_timeout_seconds", 15))
        except RewriteError as e:
            errors.append(e.to_dict())
    keywords = rewrite.keywords if rewrite else query
    meaning = rewrite.intent if rewrite else query
    ranker_query = f"{query} (meaning: {rewrite.intent})" if rewrite else query

    papers, source_errors = search_all(keywords, prefs | {"priorities": priorities}, field_keys, body.use_s2)
    errors += [e.to_dict() for e in source_errors]
    candidates = len(papers)
    try:
        retrieval.score_similarity(papers, meaning, prefs.get("interests", ""))
    except EmbeddingError as e:
        errors.append(e.to_dict())  # shortlist falls back to source order
    pool = papers
    papers = shortlist(papers, prefs.get("rank_at_most", 24), priorities)
    liked, disliked = library.rated_titles()
    try:
        papers = rank_with_timeout(papers, ranker_query, prefs, liked, disliked, prefs.get("rank_timeout_seconds", 120))
    except RankingError as e:
        errors.append(e.to_dict())
    papers = prioritize(papers, priorities)
    try:
        snapshots.save(query, prefs, field_keys, pool, papers, feedback_titles=liked[-15:] + disliked[-15:],
                       rewrite=rewrite.model_dump() if rewrite else None)
    except OSError as e:
        log.warning("couldn't save search snapshot: %s", e)
    shown = papers[:prefs["show_top"]]
    shown_keys = {p.key for p in shown}
    rest = [p for p in pool if p.key not in shown_keys]
    return {
        "rewrite": rewrite.model_dump() if rewrite else None,
        "candidates": candidates,
        "with_code": sum(p.has_code for p in papers),
        "code_index": pwc.status(),
        "papers": [serialize(p) for p in shown],
        # Rating a few candidates the ranker *didn't* show keeps the eval honest: otherwise
        # every label comes from the current ranker's top 10, and a strategy that surfaces
        # a paper it buried could never get credit.
        "unranked_sample": [serialize(p) for p in random.sample(rest, min(prefs.get("eval_sample", 5), len(rest)))],
        "errors": errors,
    }


@app.post("/api/summarize")
def summarize_paper(body: PaperIn) -> dict:
    paper = to_paper(body.paper)
    cached = library.get(paper.key)
    if cached and cached.get("summary") and not body.refresh:
        return {"summary": cached["summary"], "full_text": cached["full_text"]}
    prefs = load_prefs()
    try:
        text, full_text = summarize(paper, prefs, prefs["summary_template"])
    except Exception as e:
        raise HTTPException(502, f"Summary failed: {e}")
    library.upsert(paper, summary=text, full_text=full_text)
    return {"summary": text, "full_text": full_text}


@app.post("/api/similar")
def similar(body: SimilarIn) -> dict:
    paper = to_paper(body.paper)
    try:
        found = graph.similar(paper.to_dict(), body.key or paper.key)
    except EmbeddingError as e:
        raise HTTPException(502, str(e))
    return {"similar": found}


@app.post("/api/map")
def paper_map(body: MapIn) -> dict:
    entries = library.entries()
    ratings = {e["key"]: e["rating"] for e in entries}
    try:
        data = graph.neighbourhood(body.keys + [e["key"] for e in entries])
    except EmbeddingError as e:
        raise HTTPException(502, str(e))
    for n in data["nodes"]:
        n["rating"] = ratings.get(n["key"], 0)
        n["on_screen"] = n["key"] in body.keys
    return data


@app.post("/api/rate")
def rate(body: RateIn) -> dict:
    library.upsert(to_paper(body.paper), rating=max(-1, min(1, body.rating)))
    return {"ok": True}


@app.post("/api/save")
def save(body: SaveIn) -> dict:
    library.upsert(to_paper(body.paper), saved=body.saved)
    return {"ok": True}


@app.get("/api/library")
def get_library() -> dict:
    return {"entries": [e | {"bibtex": to_paper(e["paper"]).bibtex()} for e in library.entries()]}


@app.post("/api/library/remove")
def remove(body: KeyIn) -> dict:
    library.remove(body.key)
    return {"ok": True}
