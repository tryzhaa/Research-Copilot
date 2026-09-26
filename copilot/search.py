"""Fan out to every source for every selected field, then dedupe and hard-filter."""
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from itertools import zip_longest

from . import pwc
from .errors import SourceFetchError, SourceTimeout
from .models import Paper
from .sources import search_arxiv, search_hf_papers, search_openalex


def search_all(query: str, prefs: dict, field_keys: list[str]) -> tuple[list[Paper], list[SourceFetchError]]:
    """Returns (papers, errors). A failing source is reported, not fatal."""
    n = prefs["candidates_per_source"]
    min_year = prefs["filters"]["min_year"]
    jobs: list[tuple[str, str, Callable[[], list[Paper]]]] = []
    for key in field_keys:
        f = prefs["fields"][key]
        jobs.append(("arXiv", key, partial(search_arxiv, query, f.get("arxiv_categories", []), n, key)))
        jobs.append(("OpenAlex", key, partial(search_openalex, query, f.get("openalex_field"), n, key, min_year,
                                                     f.get("openalex_subfields"))))

    # Hugging Face Papers (Papers with Code's successor) isn't split by field, so it runs once.
    hf_field = "ml" if "ml" in field_keys else field_keys[0]
    jobs.insert(0, ("Hugging Face", hf_field, partial(search_hf_papers, query, n, hf_field)))

    # One deadline for the whole fan-out: OpenAlex alone can take 10-50 s per query on its side,
    # and a search shouldn't wait on its slowest source. Stragglers are reported and skipped;
    # their requests finish in the background and are ignored.
    deadline = prefs.get("source_timeout_seconds", 25)
    results: list[list[Paper]] = []
    errors: list[SourceFetchError] = []
    pool = ThreadPoolExecutor(max_workers=8)
    futures = [(name, key, pool.submit(fn)) for name, key, fn in jobs]
    wait([f for _, _, f in futures], timeout=deadline)
    for name, key, fut in futures:
        if not fut.done():
            errors.append(SourceFetchError(name, key, SourceTimeout(deadline)))
            continue
        try:
            results.append(fut.result())
        except Exception as e:
            errors.append(SourceFetchError(name, key, e))
    pool.shutdown(wait=False, cancel_futures=True)

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


DEFAULT_TIER_ORDER = ["datasets", "code", "cpu"]


def priority_tier(p: Paper, priorities: dict) -> int:
    """Hard preferences in tier_order (default: enough datasets > has code > no GPU needed).
    Each earlier preference outweighs all later ones combined. Higher tier sorts first."""
    met = {
        "datasets": len(p.datasets) >= priorities.get("min_datasets", 2),
        "code": priorities.get("code_first", True) and p.has_code,
        "cpu": priorities.get("prefer_cpu", True) and p.needs_gpu is False,
    }
    order = priorities.get("tier_order", DEFAULT_TIER_ORDER)
    return sum(2 ** (len(order) - 1 - i) for i, name in enumerate(order) if met.get(name))


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


def is_relevant(p: Paper, priorities: dict, sim_range: tuple[float, float]) -> bool:
    """Relevant enough for the tiers to apply: LLM relevance >= min_relevance, or when the LLM
    didn't score it, scaled similarity >= min_relevance (the top half of the search, by default)."""
    rel = p.score if p.score is not None else signals(p, sim_range)["similarity"]
    return rel is not None and rel >= priorities.get("min_relevance", 5)


def rank_value(p: Paper, priorities: dict, sim_range: tuple[float, float]) -> float:
    """Where a paper sorts, as one number (higher first): relevant, then its tier, then the
    blended score. The same order as the (relevant, tier, blend) tuple: the blend is at most 10
    and each tier step is 100, so nothing lower can outweigh something higher."""
    relevant = is_relevant(p, priorities, sim_range)
    tier = priority_tier(p, priorities) if relevant else 0
    return 1000 * relevant + 100 * tier + blended_score(p, priorities, sim_range)


def prioritize(papers: list[Paper], priorities: dict) -> list[Paper]:
    """Relevant papers first. Among them: the tier_order tiers, then the blended score.
    Tiers never lift an off-topic paper above an on-topic one (set min_relevance: 0 to allow it)."""
    rng = similarity_range(papers)
    return sorted(papers, key=lambda p: rank_value(p, priorities, rng), reverse=True)


def shortlist(papers: list[Paper], limit: int, priorities: dict) -> list[Paper]:
    """Pick which candidates the LLM reads: the most similar to query + interests. When code
    comes first (`code` leads tier_order), relevant papers with code (similarity in the
    min_relevance range) take the slots first, so the ones ranked first get read, with datasets
    and a recruiter score; an off-topic paper never buys a slot with code.
    Without similarity scores, source order holds."""
    rng = similarity_range(papers)
    code_leads = (priorities.get("tier_order", DEFAULT_TIER_ORDER) or [None])[0] == "code" \
        and priorities.get("code_first", True)

    def key(p: Paper) -> tuple[bool, float]:
        sim = p.similarity if p.similarity is not None else float("-inf")
        return (code_leads and p.has_code and is_relevant(p, priorities, rng), sim)
    return sorted(papers, key=key, reverse=True)[:limit]  # stable, so ties keep source order
