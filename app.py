"""Research Copilot web app. Run: uvicorn app:app --reload  →  http://localhost:8000"""
import hashlib
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import random
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from copilot import demo, graph, library, pwc, retrieval, snapshots, trending
from copilot.errors import EmbeddingError, RankingError, RewriteError, SourceFetchError
from copilot.llm import QueryRewrite, rank_with_timeout, rewrite_query, summarize
from copilot.models import InvalidPaper, Paper
from copilot.prefs import load_prefs
from copilot.search import attach_code, prioritize, rank_value, search_all, shortlist, similarity_range

ROOT = Path(__file__).parent
log = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    pwc.build_in_background()  # one-time Papers with Code index; searches work without it meanwhile
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.exception_handler(InvalidPaper)
async def invalid_paper(_: Request, e: InvalidPaper) -> JSONResponse:
    return JSONResponse({"detail": f"Invalid paper: {e}"}, status_code=422)


def to_paper(d: dict) -> Paper:
    return Paper.from_dict(d)


def client_ip(request: Request) -> str:
    # Hosts like Hugging Face Spaces sit behind a proxy; the visitor is the first forwarded address.
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def owner_only() -> None:
    """Routes that change the shared library. The public demo is read-only."""
    if demo.enabled():
        raise HTTPException(403, "This is a read-only demo. Run it yourself to rate and save papers: "
                                 "github.com/tryzhaa/Research-Copilot")


def spend(request: Request, kind: str) -> None:
    """In the demo, count a search or summary against the visitor's and the day's LLM quota."""
    if demo.enabled() and (refusal := demo.spend(kind, client_ip(request))):
        raise HTTPException(429, refusal)


# The owner's private signals: how the ranking judges a paper *for you* (can you run it on a CPU,
# would it impress a recruiter, how close it is to your interests). The public demo still ranks
# with them but blanks them in every response, so visitors never receive them, not just never see them.
PRIVATE_FIELDS: dict[str, object] = {"needs_gpu": None, "compute_note": "", "recruiter": None,
                                     "recruiter_reason": "", "similarity": None, "preference": None}


def public(d: dict) -> dict:
    """A paper dict as the current viewer may see it: unchanged locally, private signals blanked in the demo."""
    if not demo.enabled():
        return d
    return d | {k: v for k, v in PRIVATE_FIELDS.items() if k in d}


def serialize(p: Paper) -> dict:
    entry = library.get(p.key) or {}
    return public(p.to_dict()) | {
        "key": p.key,
        "has_code": p.has_code,
        "dataset_links": pwc.dataset_links(p.datasets),
        "bibtex": p.bibtex(),
        "rating": entry.get("rating", 0),
        "saved": entry.get("saved", False),
        "folders": entry.get("folders", []),
    }


class SearchIn(BaseModel):
    query: str
    fields: list[str]
    code_only: bool = False
    # Demo only: the visitor's recent liked / disliked titles, from the library in their browser,
    # so their ratings steer their own ranking the way yours do locally.
    liked: list[str] = []
    disliked: list[str] = []


def visitor_titles(titles: list[str]) -> list[str]:
    """What a visitor may send the ranker: their 15 most recent, each trimmed."""
    return [t.strip()[:300] for t in titles[-15:] if isinstance(t, str) and t.strip()]


class PaperIn(BaseModel):
    paper: dict
    refresh: bool = False


class RateIn(BaseModel):
    paper: dict
    rating: int  # -1, 0, 1


class SaveIn(BaseModel):
    paper: dict
    saved: bool


class FolderIn(BaseModel):
    paper: dict
    folder: str
    add: bool  # false = take it out of the folder


class KeyIn(BaseModel):
    key: str


class SimilarIn(BaseModel):
    paper: dict
    key: str | None = None  # the map sends graph keys directly


class MapIn(BaseModel):
    keys: list[str]               # papers on screen
    include_library: bool = True  # the map tab adds your library; the graph beside results doesn't
    limit: int = 250
    related: int = 0              # also return this many similar papers you haven't rated or saved
    # Demo only: the visitor's library, from their browser (the server keeps none of it).
    papers: list[dict] = []       # their saved and rated papers, to place on the map
    ratings: dict[str, int] = {}  # key -> -1 / 1


def versioned(match: re.Match[str]) -> str:
    """/static/icon.svg → /static/icon.svg?v=<content hash>: a new address whenever the file
    changes, so browsers never keep an old copy. Browsers hold on to a site icon especially
    long, past normal refreshes."""
    path = ROOT / "static" / match.group(1)
    if not path.is_file():
        return match.group(0)
    return f"/static/{match.group(1)}?v={hashlib.sha256(path.read_bytes()).hexdigest()[:10]}"


@app.get("/")
def index() -> HTMLResponse:
    html = (ROOT / "static" / "index.html").read_text()
    return HTMLResponse(re.sub(r"/static/([\w.-]+)(?=\")", versioned, html))


@app.get("/api/prefs")
def get_prefs() -> dict:
    prefs = load_prefs()
    return {
        "fields": {k: v["label"] for k, v in prefs["fields"].items()},
        "provider": prefs.get("provider", "ollama"),
        "model": prefs["model"],
        "effort": prefs.get("effort", "medium"),
        "demo": demo.limits() if demo.enabled() else None,
    }


@app.get("/api/trending")
def get_trending() -> dict:
    """The home screen's list before any search. No LLM, so no demo quota."""
    try:
        papers, fetched_at, note = trending.trending()
    except SourceFetchError as e:
        raise HTTPException(502, str(e))
    return {"papers": [serialize(p) for p in papers], "fetched_at": fetched_at, "note": note}


@app.post("/api/search")
def search(body: SearchIn, request: Request) -> dict:
    prefs = load_prefs()
    field_keys = [k for k in body.fields if k in prefs["fields"]]
    if not body.query.strip() or not field_keys:
        raise HTTPException(400, "Enter a query and pick at least one field.")
    spend(request, "search")
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

    papers, source_errors = search_all(keywords, prefs | {"priorities": priorities}, field_keys)
    errors += [e.to_dict() for e in source_errors]
    candidates = len(papers)
    try:
        retrieval.score_similarity(papers, meaning, prefs.get("interests", ""))
    except EmbeddingError as e:
        errors.append(e.to_dict())  # shortlist falls back to source order
    pool = papers
    papers = shortlist(papers, prefs.get("rank_at_most", 24), priorities)
    liked, disliked = ((visitor_titles(body.liked), visitor_titles(body.disliked)) if demo.enabled()
                       else library.rated_titles())
    try:
        papers = rank_with_timeout(papers, ranker_query, prefs, liked, disliked, prefs.get("rank_timeout_seconds", 120))
    except RankingError as e:
        errors.append(e.to_dict())
    try:
        retrieval.score_preference(papers)  # after ranking: it reads the LLM's scores too
    except EmbeddingError as e:
        errors.append(e.to_dict())
    papers = prioritize(papers, priorities)
    search_id = None
    try:
        search_id = snapshots.search_id(snapshots.save(
            query, prefs, field_keys, pool, papers, feedback_titles=liked[-15:] + disliked[-15:],
            rewrite=rewrite.model_dump() if rewrite else None))
    except OSError as e:
        log.warning("couldn't save search snapshot: %s", e)
    rng = similarity_range(papers)
    shown = papers[:prefs["show_top"]]
    shown_keys = {p.key for p in shown}
    rest = [p for p in pool if p.key not in shown_keys]
    return {
        "rewrite": rewrite.model_dump() if rewrite else None,
        "candidates": candidates,
        "with_code": sum(p.has_code for p in papers),
        "code_index": pwc.status(),
        "search_id": search_id,  # to add a paper from the map into these results later (/api/place)
        # rank: where each sorts, so a paper added from the map can be slotted in among them
        "papers": [serialize(p) | {"rank": round(rank_value(p, priorities, rng), 4)} for p in shown],
        # Rating a few candidates the ranker *didn't* show keeps the eval honest: otherwise
        # every label comes from the current ranker's top 10, and a strategy that surfaces
        # a paper it buried could never get credit.
        # (Not in the demo, where nobody can rate.)
        "unranked_sample": [] if demo.enabled() else
                           [serialize(p) for p in random.sample(rest, min(prefs.get("eval_sample", 5), len(rest)))],
        "errors": errors,
    }


@app.post("/api/summarize")
def summarize_paper(body: PaperIn, request: Request) -> dict:
    paper = to_paper(body.paper)
    cached = library.get(paper.key)
    # The demo always serves a cached summary: regenerating is a paid call for no new paper.
    if cached and cached.get("summary") and (not body.refresh or demo.enabled()):
        return {"summary": cached["summary"], "full_text": cached["full_text"]}
    spend(request, "summary")
    prefs = load_prefs()
    try:
        text, full_text = summarize(paper, prefs, prefs["summary_template"])
    except Exception as e:
        raise HTTPException(502, f"Summary failed: {e}")
    library.upsert(paper, summary=text, full_text=full_text)
    return {"summary": text, "full_text": full_text}


class PlaceIn(BaseModel):
    search_id: str
    key: str
    paper: dict = {}              # what the map knows of it: used when nothing fuller is saved
    liked: list[str] = []         # the demo visitor's ratings, as for /api/search
    disliked: list[str] = []


@app.post("/api/place")
def place(body: PlaceIn, request: Request) -> dict:
    """A paper clicked on the map, made a full result of the search on screen: rebuilt from its
    fullest saved record, its code looked up, its similarity measured against this search, read by
    the model for this query (relevance, datasets, recruiter score, GPU needs), and given the same
    rank value as the results, so the page can slot it in where the ranking puts it. If the model
    is busy, it keeps its stored signals and ranks on similarity instead of relevance."""
    prefs = load_prefs()
    snap = snapshots.load(body.search_id)
    if snap is None:
        raise HTTPException(404, "That search is no longer saved. Search again to add papers from the map.")
    stored = snapshots.find_paper(body.key) or ({"title": body.paper.get("title")} | body.paper
                                                if body.paper.get("title") else None)
    if stored is None:
        raise HTTPException(404, "No record of that paper.")
    spend(request, "place")
    p = Paper.from_dict(stored)
    p.score, p.reason = None, ""  # relevance was to another search's query
    priorities = prefs.get("priorities", {})
    errors: list[dict] = []
    attach_code([p])
    intent = (snap.get("rewrite") or {}).get("intent")
    try:
        retrieval.score_similarity([p], intent or snap["query"], snap.get("interests", ""))
    except EmbeddingError as e:
        errors.append(e.to_dict())
    liked, disliked = ((visitor_titles(body.liked), visitor_titles(body.disliked)) if demo.enabled()
                       else library.rated_titles())
    ranker_query = f"{snap['query']} (meaning: {intent})" if intent else snap["query"]
    try:
        [p] = rank_with_timeout([p], ranker_query, prefs, liked, disliked, prefs.get("place_timeout_seconds", 75))
    except RankingError as e:
        errors.append(e.to_dict())
    try:
        retrieval.score_preference([p])
    except EmbeddingError as e:
        errors.append(e.to_dict())
    # The search's own range: similarity is scaled within the papers the results were ranked among.
    ranked = [c for c in snap["candidates"] if c.get("shortlisted")] or snap["candidates"]
    rng = similarity_range([Paper(title=c["title"], similarity=c.get("similarity")) for c in ranked])
    return {"paper": serialize(p) | {"rank": round(rank_value(p, priorities, rng), 4)}, "errors": errors}


@app.post("/api/similar")
def similar(body: SimilarIn) -> dict:
    paper = to_paper(body.paper)
    try:
        found = graph.similar(paper.to_dict(), body.key or paper.key)
    except EmbeddingError as e:
        raise HTTPException(502, str(e))
    items = [public(s) for s in found]
    if demo.enabled():
        for s in items:
            s.pop("score", None)  # the graph walk's rank for this paper: private, like similarity
    return {"similar": items}


@app.post("/api/map")
def paper_map(body: MapIn) -> dict:
    if demo.enabled():
        # The visitor's library comes with the request: their browser holds it, the server doesn't.
        visitor = {p["key"]: p for p in body.papers[:300] if isinstance(p.get("key"), str)}
        ratings = {k: max(-1, min(1, v)) for k, v in list(body.ratings.items())[:1000]}
        library_keys, known = list(visitor), set(visitor) | set(ratings)
    else:
        entries = library.entries()
        ratings = {e["key"]: e["rating"] for e in entries}
        library_keys = [e["key"] for e in entries]
        known = {e["key"] for e in entries if e.get("rating") or e.get("saved")}
        visitor = {}
    try:
        g = graph.with_papers(visitor) if visitor else None
        focus = body.keys + (library_keys if body.include_library else [])
        data = graph.neighbourhood(focus, limit=min(body.limit, 400), g=g)
        found = graph.related(body.keys, exclude=known, n=min(body.related, 30), g=g) if body.related else []
    except EmbeddingError as e:
        raise HTTPException(502, str(e))
    for n in data["nodes"]:
        n["rating"] = ratings.get(n["key"], 0)
        n["on_screen"] = n["key"] in body.keys
    if body.related:
        # The library's map lists these under it, as rows you can like or save.
        data["related"] = [serialize(Paper.from_dict(p)) | {"via": p["via"]} | ({} if demo.enabled() else {"sim": p["sim"]})
                           for p in found]
    return data


@app.post("/api/rate")
def rate(body: RateIn, _: None = Depends(owner_only)) -> dict:
    library.upsert(to_paper(body.paper), rating=max(-1, min(1, body.rating)))
    return {"ok": True}


@app.post("/api/save")
def save(body: SaveIn, _: None = Depends(owner_only)) -> dict:
    # Folders are part of saving (filing a paper saves it), so unsaving takes it out of them too.
    changes: dict = {"saved": True} if body.saved else {"saved": False, "folders": []}
    entry = library.upsert(to_paper(body.paper), **changes)
    return {"saved": entry["saved"], "folders": entry["folders"], "all": library.folders()}


@app.post("/api/folder")
def folder(body: FolderIn, _: None = Depends(owner_only)) -> dict:
    try:
        entry = library.set_folder(to_paper(body.paper), body.folder, body.add)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"folders": entry["folders"], "saved": entry["saved"], "all": library.folders()}


@app.get("/api/folders")
def get_folders() -> dict:
    return {"folders": [] if demo.enabled() else library.folders()}  # visitors' folders live in their browser


@app.get("/api/library")
def get_library() -> dict:
    if demo.enabled():
        return {"entries": []}  # a visitor's library lives in their browser, not on the server
    return {"entries": [e | {"bibtex": to_paper(e["paper"]).bibtex()} for e in library.entries()]}


@app.post("/api/library/remove")
def remove(body: KeyIn, _: None = Depends(owner_only)) -> dict:
    library.remove(body.key)
    return {"ok": True}
