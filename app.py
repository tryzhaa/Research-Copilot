"""Research Copilot web app. Run: uvicorn app:app --reload  →  http://localhost:8000"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import fields
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from copilot import library, pwc
from copilot.errors import RankingError
from copilot.llm import rank, summarize
from copilot.models import Paper
from copilot.prefs import load_prefs
from copilot.search import prioritize, search_all, shortlist

ROOT = Path(__file__).parent
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
    papers, source_errors = search_all(body.query.strip(), prefs | {"priorities": priorities}, field_keys, body.use_s2)
    errors = [e.to_dict() for e in source_errors]
    candidates = len(papers)
    papers = shortlist(papers, prefs.get("rank_at_most", 24), priorities)
    liked, disliked = library.rated_titles()
    try:
        papers = rank(papers, body.query, prefs, liked, disliked)
    except Exception as e:
        errors.append(RankingError(f"ranking failed, showing unranked results: {e}").to_dict())
    papers = prioritize(papers, priorities)
    return {
        "candidates": candidates,
        "with_code": sum(p.has_code for p in papers),
        "code_index": pwc.status(),
        "papers": [serialize(p) for p in papers[:prefs["show_top"]]],
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
