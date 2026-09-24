"""Fetchers for arXiv, OpenAlex and Semantic Scholar. Each returns list[Paper]."""
import os
import re
import time

import feedparser
import httpx

from .models import Paper

HEADERS = {"User-Agent": "research-copilot/0.1"}
TIMEOUT = 20
STOPWORDS = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "by", "from", "via", "or", "is", "are"}


def _get(url: str, params: dict, headers: dict = HEADERS, tries: int = 4) -> httpx.Response:
    """GET with backoff on rate limits and server hiccups — parallel field searches trip them easily."""
    for attempt in range(tries):
        r = httpx.get(url, params=params, headers=headers, timeout=TIMEOUT)
        if r.status_code not in (429, 500, 502, 503) or attempt == tries - 1:
            r.raise_for_status()
            return r
        time.sleep(float(r.headers.get("Retry-After", 0)) or 1.5 * 2 ** attempt)


def search_arxiv(query: str, categories: list[str], limit: int, field: str) -> list[Paper]:
    if not categories:
        return []
    cats = " OR ".join(f"cat:{c}" for c in categories)
    # arXiv applies a field prefix to one word only, so every term needs its own `all:`.
    terms = [t for t in re.findall(r"[\w-]+", query.lower()) if t not in STOPWORDS] or [query]
    params = {
        "search_query": " AND ".join(f"all:{t}" for t in terms) + f" AND ({cats})",
        "max_results": limit,
        "sortBy": "relevance",
    }
    r = _get("https://export.arxiv.org/api/query", params)
    papers = []
    for e in feedparser.parse(r.text).entries:
        arxiv_id = e.id.rsplit("/abs/", 1)[-1]
        papers.append(Paper(
            title=" ".join(e.title.split()),
            abstract=" ".join(e.summary.split()),
            authors=[a.name for a in e.get("authors", [])],
            year=int(e.published[:4]),
            venue="arXiv",
            doi=e.get("arxiv_doi", ""),
            arxiv_id=arxiv_id,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            source="arXiv",
            field=field,
        ))
    return papers


def search_hf_papers(query: str, limit: int, field: str) -> list[Paper]:
    """Hugging Face Papers — where paperswithcode.com now redirects. Live, with GitHub repos and stars."""
    r = _get("https://huggingface.co/api/papers/search", {"q": query, "limit": limit})
    papers = []
    for hit in r.json():
        p = hit.get("paper") or {}
        arxiv_id = p.get("id", "")
        if not arxiv_id:
            continue
        papers.append(Paper(
            title=" ".join((p.get("title") or "").split()),
            abstract=" ".join((p.get("summary") or "").split()),
            authors=[a["name"] for a in (p.get("authors") or [])[:12] if a.get("name")],
            year=int(p["publishedAt"][:4]) if p.get("publishedAt") else None,
            venue="arXiv",
            arxiv_id=arxiv_id,
            url=f"https://huggingface.co/papers/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            code_url=p.get("githubRepo") or "",
            stars=p.get("githubStars") or 0,
            source="Hugging Face",
            field=field,
        ))
    return papers


def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def search_openalex(query: str, field_id: int | None, limit: int, field: str, min_year: int) -> list[Paper]:
    if not field_id:
        return []
    params = {
        "search": query,
        "filter": f"primary_topic.field.id:{field_id},from_publication_date:{min_year}-01-01",
        "per-page": limit,
    }
    if email := os.getenv("OPENALEX_EMAIL"):
        params["mailto"] = email  # polite pool: faster, more reliable
    r = _get("https://api.openalex.org/works", params)
    papers = []
    for w in r.json().get("results", []):
        loc = w.get("primary_location") or {}
        oa = w.get("best_oa_location") or {}
        papers.append(Paper(
            title=w.get("title") or "",
            abstract=_openalex_abstract(w.get("abstract_inverted_index")),
            authors=[a["author"]["display_name"] for a in w.get("authorships", [])[:12]],
            year=w.get("publication_year"),
            venue=((loc.get("source") or {}).get("display_name")) or "",
            doi=(w.get("doi") or "").replace("https://doi.org/", ""),
            url=w.get("doi") or w.get("id", ""),
            pdf_url=oa.get("pdf_url") or "",
            citations=w.get("cited_by_count", 0),
            source="OpenAlex",
            field=field,
        ))
    return papers


def search_semantic_scholar(query: str, s2_field: str | None, limit: int, field: str, min_year: int) -> list[Paper]:
    """Optional: works without a key but is heavily rate-limited. Set S2_API_KEY for reliability."""
    if not s2_field:
        return []
    headers = dict(HEADERS)
    if key := os.getenv("S2_API_KEY"):
        headers["x-api-key"] = key
    params = {
        "query": query,
        "fieldsOfStudy": s2_field,
        "year": f"{min_year}-",
        "limit": limit,
        "fields": "title,abstract,authors,year,venue,externalIds,url,openAccessPdf,citationCount",
    }
    r = _get("https://api.semanticscholar.org/graph/v1/paper/search", params, headers)
    papers = []
    for p in r.json().get("data", []):
        ids = p.get("externalIds") or {}
        papers.append(Paper(
            title=p.get("title") or "",
            abstract=p.get("abstract") or "",
            authors=[a["name"] for a in (p.get("authors") or [])[:12]],
            year=p.get("year"),
            venue=p.get("venue") or "",
            doi=ids.get("DOI", ""),
            arxiv_id=ids.get("ArXiv", ""),
            url=p.get("url") or "",
            pdf_url=(p.get("openAccessPdf") or {}).get("url") or "",
            citations=p.get("citationCount") or 0,
            source="Semantic Scholar",
            field=field,
        ))
    return papers
