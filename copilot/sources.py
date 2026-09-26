"""Fetchers for arXiv, OpenAlex and Hugging Face Papers. Each returns list[Paper]."""
import os
import re
import ssl
import threading
import time
from collections import OrderedDict

import certifi
import feedparser
import httpx

from .models import Paper

HEADERS = {"User-Agent": "research-copilot/0.1"}
TIMEOUT = 20
STOPWORDS = {"a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "by", "from", "via", "or", "is", "are"}


ARXIV_GAP = 3.0  # arXiv's API terms: no more than one request every 3 seconds
_arxiv_lock = threading.Lock()
_arxiv_last = 0.0


def arxiv_wait() -> None:
    """Space requests to arXiv (API and PDFs) ARXIV_GAP apart across all threads, as arXiv's
    API terms ask; a three-field search would otherwise send three queries at once."""
    global _arxiv_last
    with _arxiv_lock:
        delay = _arxiv_last + ARXIV_GAP - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _arxiv_last = time.monotonic()


def _arxiv_tls() -> ssl.SSLContext:
    """TLS for arXiv: offer the classic X25519 key exchange only.

    OpenSSL 3.5 (e.g. python:3.12-slim) offers the post-quantum X25519MLKEM768 key share by
    default, and arXiv's edge answers those connections with an empty 406: every arXiv search
    and PDF failed in Docker while the same request from a Mac on OpenSSL 3.0 worked. Pinning
    the key exchange fixed it (406 -> 200) without touching TLS for any other host."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.set_ecdh_curve("X25519")
    return ctx


ARXIV_TLS = _arxiv_tls()


def tls_for(url: str) -> ssl.SSLContext | bool:
    """The `verify=` argument for an httpx request to `url`."""
    return ARXIV_TLS if "arxiv.org" in url else True


# Shared by every search in the process: successful responses are kept for CACHE_TTL, and
# simultaneous identical requests wait for the first one instead of each hitting the source.
# Several visitors trying the same query cost one request per source, which matters most for
# arXiv, where every request in the process shares one 3-second slot (arxiv_wait).
CACHE_TTL = 600.0
CACHE_BYTES = 16 * 1024 * 1024
_cache: "OrderedDict[tuple, tuple[float, httpx.Response]]" = OrderedDict()
_cache_size = 0
_cache_lock = threading.Lock()
_inflight: dict[tuple, threading.Lock] = {}


def clear_cache() -> None:
    global _cache_size
    with _cache_lock:
        _cache.clear()
        _cache_size = 0


def _cached(key: tuple) -> httpx.Response | None:
    """Call with _cache_lock held."""
    hit = _cache.get(key)
    if hit is None:
        return None
    if time.monotonic() - hit[0] > CACHE_TTL:
        _evict(key)
        return None
    _cache.move_to_end(key)
    return hit[1]


def _evict(key: tuple) -> None:
    global _cache_size
    _, r = _cache.pop(key)
    _cache_size -= len(r.content)


def _store(key: tuple, r: httpx.Response) -> None:
    global _cache_size
    if key in _cache:
        _evict(key)
    _cache[key] = (time.monotonic(), r)
    _cache_size += len(r.content)
    while _cache_size > CACHE_BYTES and len(_cache) > 1:
        _evict(next(iter(_cache)))  # least recently used first


def _get(url: str, params: dict, headers: dict = HEADERS, tries: int = 4) -> httpx.Response:
    """GET through the shared cache; concurrent identical requests share one fetch."""
    key = (url, tuple(sorted((k, str(v)) for k, v in params.items())))
    with _cache_lock:
        if (hit := _cached(key)) is not None:
            return hit
        flight = _inflight.setdefault(key, threading.Lock())
    with flight:
        with _cache_lock:  # the request we waited on may have filled it
            if (hit := _cached(key)) is not None:
                return hit
        try:
            r = _fetch(url, params, headers, tries)
            with _cache_lock:
                _store(key, r)
            return r
        finally:
            with _cache_lock:
                _inflight.pop(key, None)


def _fetch(url: str, params: dict, headers: dict, tries: int) -> httpx.Response:
    """GET with backoff on rate limits and server hiccups — parallel field searches trip them easily."""
    for attempt in range(tries):
        if "arxiv.org" in url:
            arxiv_wait()
        r = httpx.get(url, params=params, headers=headers, timeout=TIMEOUT, verify=tls_for(url))
        if r.status_code not in (429, 500, 502, 503) or attempt == tries - 1:
            r.raise_for_status()
            return r
        time.sleep(float(r.headers.get("Retry-After", 0)) or 1.5 * 2 ** attempt)
    raise ValueError("tries must be at least 1")


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


def _hf_paper(p: dict, field: str) -> Paper | None:
    """A Hugging Face Papers record (search hit or daily/trending entry) as a Paper."""
    arxiv_id = p.get("id", "")
    if not arxiv_id:
        return None
    return Paper(
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
        upvotes=p.get("upvotes") or 0,
        tldr=" ".join((p.get("ai_summary") or "").split()),
        source="Hugging Face",
        field=field,
    )


def search_hf_papers(query: str, limit: int, field: str) -> list[Paper]:
    """Hugging Face Papers — where paperswithcode.com now redirects. Live, with GitHub repos and stars."""
    r = _get("https://huggingface.co/api/papers/search", {"q": query, "limit": limit})
    return [p for hit in r.json() if (p := _hf_paper(hit.get("paper") or {}, field))]


def hf_trending(limit: int) -> list[Paper]:
    """Hugging Face's trending papers, the list on huggingface.co/papers/trending."""
    # Straight to the source: trending.py keeps its own hour-long cache of this list.
    r = _fetch("https://huggingface.co/api/daily_papers", {"sort": "trending", "limit": limit}, HEADERS, 4)
    return [p for hit in r.json() if (p := _hf_paper(hit.get("paper") or {}, "ml"))]


def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def search_openalex(query: str, field_id: int | None, limit: int, field: str, min_year: int,
                    subfields: list[int] | None = None) -> list[Paper]:
    """Scoped to a whole OpenAlex field, or to specific subfields (which may span fields) when given."""
    if subfields:
        scope = "primary_topic.subfield.id:" + "|".join(map(str, subfields))
    elif field_id:
        scope = f"primary_topic.field.id:{field_id}"
    else:
        return []
    params = {
        "search": query,
        "filter": f"{scope},from_publication_date:{min_year}-01-01",
        "per-page": limit,
    }
    if key := os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = key  # free key from openalex.org: no anonymous rate limiting
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
