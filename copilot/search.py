"""Fan out to every source for every selected field, then dedupe and hard-filter."""
from concurrent.futures import ThreadPoolExecutor

from .models import Paper
from .sources import search_arxiv, search_openalex, search_semantic_scholar


def search_all(query: str, prefs: dict, field_keys: list[str], use_s2: bool = False) -> tuple[list[Paper], list[str]]:
    """Returns (papers, errors). A failing source is reported, not fatal."""
    n = prefs["candidates_per_source"]
    min_year = prefs["filters"]["min_year"]
    jobs = []
    for key in field_keys:
        f = prefs["fields"][key]
        jobs.append(("arXiv", key, lambda f=f, key=key: search_arxiv(query, f.get("arxiv_categories", []), n, key)))
        jobs.append(("OpenAlex", key, lambda f=f, key=key: search_openalex(query, f.get("openalex_field"), n, key, min_year)))
        if use_s2:
            jobs.append(("Semantic Scholar", key, lambda f=f, key=key: search_semantic_scholar(query, f.get("s2_field"), n, key, min_year)))

    papers, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [(name, key, pool.submit(fn)) for name, key, fn in jobs]
        for name, key, fut in futures:
            try:
                papers.extend(fut.result())
            except Exception as e:
                errors.append(f"{name} ({key}): {e}")

    return apply_filters(dedupe(papers), prefs["filters"]), errors


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Keep one copy per paper, preferring the record with the most metadata."""
    merged: list[Paper] = []
    by_id: dict[str, Paper] = {}
    for p in papers:
        cur = next((by_id[i] for i in p.ids if i in by_id), None)
        if cur is None:
            merged.append(p)
            cur = p
        else:
            # Merge: fill gaps in the existing record from the duplicate.
            cur.abstract = cur.abstract or p.abstract
            cur.pdf_url = cur.pdf_url or p.pdf_url
            cur.arxiv_id = cur.arxiv_id or p.arxiv_id
            if not cur.doi or cur.doi.lower().startswith("10.48550/"):
                cur.doi = p.doi or cur.doi
            cur.citations = max(cur.citations, p.citations)
            if p.venue and cur.venue in ("", "arXiv"):
                cur.venue = p.venue
        for i in p.ids + cur.ids:
            by_id.setdefault(i, cur)
    return merged


def apply_filters(papers: list[Paper], filters: dict) -> list[Paper]:
    excluded = [k.lower() for k in filters.get("exclude_keywords", [])]
    out = []
    for p in papers:
        if not p.title:
            continue
        if p.year and p.year < filters.get("min_year", 0):
            continue
        if p.citations < filters.get("min_citations", 0) and p.source != "arXiv":
            continue  # new preprints have no citations yet; don't punish them
        text = (p.title + " " + p.abstract).lower()
        if any(k in text for k in excluded):
            continue
        out.append(p)
    return out
