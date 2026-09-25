"""Fan out to every source for every selected field, then dedupe and hard-filter."""
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import zip_longest

from . import pwc
from .errors import SourceFetchError
from .models import Paper
from .sources import search_arxiv, search_hf_papers, search_openalex, search_semantic_scholar


def search_all(query: str, prefs: dict, field_keys: list[str], use_s2: bool = False) -> tuple[list[Paper], list[SourceFetchError]]:
    """Returns (papers, errors). A failing source is reported, not fatal."""
    n = prefs["candidates_per_source"]
    min_year = prefs["filters"]["min_year"]
    jobs: list[tuple[str, str, Callable[[], list[Paper]]]] = []
    for key in field_keys:
        f = prefs["fields"][key]
        jobs.append(("arXiv", key, partial(search_arxiv, query, f.get("arxiv_categories", []), n, key)))
        jobs.append(("OpenAlex", key, partial(search_openalex, query, f.get("openalex_field"), n, key, min_year,
                                                     f.get("openalex_subfields"))))
        if use_s2:
            jobs.append(("Semantic Scholar", key, partial(search_semantic_scholar, query, f.get("s2_field"), n, key, min_year)))

    # Hugging Face Papers (Papers with Code's successor) isn't split by field, so it runs once.
    hf_field = "ml" if "ml" in field_keys else field_keys[0]
    jobs.insert(0, ("Hugging Face", hf_field, partial(search_hf_papers, query, n, hf_field)))

    results: list[list[Paper]] = []
    errors: list[SourceFetchError] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [(name, key, pool.submit(fn)) for name, key, fn in jobs]
        for name, key, fut in futures:
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(SourceFetchError(name, key, e))

    # Round-robin across sources so list order roughly tracks each source's own relevance order.
    papers = [p for rank in zip_longest(*results) for p in rank if p is not None]

    papers = dedupe(papers)
    attach_code(papers)
    return apply_filters(papers, prefs["filters"], prefs.get("priorities", {})), errors


def _arxiv_id(p: Paper) -> str:
    return next((i.removeprefix("arxiv:") for i in p.ids if i.startswith("arxiv:")), "")


def attach_code(papers: list[Paper]) -> None:
    """Fill code links from the local Papers with Code index. Keeps a repo a live source already gave."""
    found = pwc.lookup([_arxiv_id(p) for p in papers])
    for p in papers:
        hit = found.get(_arxiv_id(p))
        if not hit:
            continue
        if not p.code_url or hit["official"]:
            p.code_url = hit["repo"]
            p.code_official = hit["official"]
        p.code_framework = hit["framework"] if hit["framework"] != "none" else p.code_framework


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
            if p.code_url and not cur.code_url:
                cur.code_url, cur.code_official, cur.code_framework = p.code_url, p.code_official, p.code_framework
            cur.stars = max(cur.stars, p.stars)
            if p.venue and cur.venue in ("", "arXiv"):
                cur.venue = p.venue
        for i in p.ids + cur.ids:
            by_id.setdefault(i, cur)
    return merged


def apply_filters(papers: list[Paper], filters: dict, priorities: dict) -> list[Paper]:
    excluded = [k.lower() for k in filters.get("exclude_keywords", [])]
    out = []
    for p in papers:
        if not p.title:
            continue
        if priorities.get("require_code") and not p.has_code:
            continue
        if p.year and p.year < filters.get("min_year", 0):
            continue
        if p.citations < filters.get("min_citations", 0) and p.source not in ("arXiv", "Hugging Face"):
            continue  # new preprints have no citations yet; don't punish them
        text = (p.title + " " + p.abstract).lower()
        if any(k in text for k in excluded):
            continue
        out.append(p)
    return out


def priority_tier(p: Paper, priorities: dict) -> int:
    """Hard preferences, in order: has code > enough datasets > no GPU needed. Higher tier sorts first."""
    tier = 0
    if priorities.get("code_first", True) and p.has_code:
        tier += 4
    if len(p.datasets) >= priorities.get("min_datasets", 2):
        tier += 2
    if priorities.get("prefer_cpu", True) and p.needs_gpu is False:
        tier += 1
    return tier


DEFAULT_WEIGHTS = {"relevance": 0.35, "recruiter": 0.35, "similarity": 0.3, "preference": 0.0}


def signals(p: Paper, sim_range: tuple[float, float]) -> dict[str, float | None]:
    """Every ranking signal on a 0-10 scale, or None when that stage didn't run for this paper.

    Similarity is min-max scaled within the search: raw cosine values from one embedding
    model bunch into a narrow band, and only their order within a search matters here.
    """
    lo, hi = sim_range
    sim = None if p.similarity is None else 10 * (p.similarity - lo) / max(1e-9, hi - lo)
    pref = None if p.preference is None else 10 * p.preference
    return {"relevance": p.score, "recruiter": p.recruiter, "similarity": sim, "preference": pref}


def blended_score(p: Paper, priorities: dict, sim_range: tuple[float, float] = (0.0, 1.0)) -> float:
    """Weighted mean of the signals this paper has. Missing ones drop out and the rest are
    renormalized, so when the LLM ranker fails or times out the order falls back to similarity."""
    w = DEFAULT_WEIGHTS | priorities.get("weights", {})
    present = {k: v for k, v in signals(p, sim_range).items() if v is not None and w.get(k, 0) > 0}
    total = sum(w[k] for k in present)
    return sum(w[k] * v for k, v in present.items()) / total if total else 0.0


def similarity_range(papers: list[Paper]) -> tuple[float, float]:
    sims = [p.similarity for p in papers if p.similarity is not None]
    return (min(sims), max(sims)) if sims else (0.0, 1.0)


def prioritize(papers: list[Paper], priorities: dict) -> list[Paper]:
    rng = similarity_range(papers)
    return sorted(papers, key=lambda p: (priority_tier(p, priorities), blended_score(p, priorities, rng)), reverse=True)


def shortlist(papers: list[Paper], limit: int, priorities: dict) -> list[Paper]:
    """Pick which candidates the LLM reads: the most similar to query + interests, papers with
    code first when code_first is on. Without similarity scores, source order holds."""
    def key(p: Paper) -> tuple[bool, float]:
        has_code = priorities.get("code_first", True) and p.has_code
        return has_code, p.similarity if p.similarity is not None else float("-inf")
    return sorted(papers, key=key, reverse=True)[:limit]  # stable, so ties keep source order
