"""Research Copilot web app. Run: uvicorn app:app --reload  →  http://localhost:8000"""
from dataclasses import fields
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from copilot import library
from copilot.llm import rank, summarize
from copilot.models import Paper
from copilot.prefs import load_prefs
from copilot.search import search_all

ROOT = Path(__file__).parent
PAPER_FIELDS = {f.name for f in fields(Paper)}

app = FastAPI()
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def to_paper(d: dict) -> Paper:
    return Paper(**{k: v for k, v in d.items() if k in PAPER_FIELDS})


def serialize(p: Paper) -> dict:
    entry = library.get(p.key) or {}
    return p.to_dict() | {
        "key": p.key,
        "bibtex": p.bibtex(),
        "rating": entry.get("rating", 0),
        "saved": entry.get("saved", False),
    }


class SearchIn(BaseModel):
    query: str
    fields: list[str]
    use_s2: bool = False


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
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/prefs")
def get_prefs():
    prefs = load_prefs()
    return {
        "fields": {k: v["label"] for k, v in prefs["fields"].items()},
        "provider": prefs.get("provider", "ollama"),
        "model": prefs["model"],
        "effort": prefs.get("effort", "medium"),
    }


@app.post("/api/search")
def search(body: SearchIn):
    prefs = load_prefs()
    field_keys = [k for k in body.fields if k in prefs["fields"]]
    if not body.query.strip() or not field_keys:
        raise HTTPException(400, "Enter a query and pick at least one field.")
    papers, errors = search_all(body.query.strip(), prefs, field_keys, body.use_s2)
    candidates = len(papers)
    liked, disliked = library.rated_titles()
    try:
        papers = rank(papers, body.query, prefs, liked, disliked)
    except Exception as e:
        errors.append(f"Ranking failed, showing unranked results: {e}")
    return {
        "candidates": candidates,
        "papers": [serialize(p) for p in papers[:prefs["show_top"]]],
        "errors": errors,
    }


@app.post("/api/summarize")
def summarize_paper(body: PaperIn):
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
def rate(body: RateIn):
    library.upsert(to_paper(body.paper), rating=max(-1, min(1, body.rating)))
    return {"ok": True}


@app.post("/api/save")
def save(body: SaveIn):
    library.upsert(to_paper(body.paper), saved=body.saved)
    return {"ok": True}


@app.get("/api/library")
def get_library():
    return {"entries": [e | {"bibtex": to_paper(e["paper"]).bibtex()} for e in library.entries()]}


@app.post("/api/library/remove")
def remove(body: KeyIn):
    library.remove(body.key)
    return {"ok": True}
