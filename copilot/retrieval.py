"""Embed → retrieve: score every candidate by semantic similarity to the query and interests."""
import numpy as np

from .embed_cache import get_embeddings
from .embeddings import embed_query, normalize, paper_text
from .models import Paper
from .preference_model import predict


def rank_by_similarity(query_vec: np.ndarray, paper_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity of each paper to the query. Both sides are unit length."""
    sims: np.ndarray = paper_vecs @ query_vec
    return sims


def query_vector(query: str, interests: str) -> np.ndarray:
    """The search is for this query *as this person*: equal parts query and standing interests."""
    q = embed_query(query)
    return normalize(q + embed_query(interests)) if interests.strip() else q


def paper_vectors(papers: list[Paper]) -> np.ndarray:
    return get_embeddings([p.key for p in papers], [paper_text(p.title, p.abstract) for p in papers])


def score_similarity(papers: list[Paper], query: str, interests: str) -> np.ndarray:
    """Sets p.similarity on every paper."""
    vecs = paper_vectors(papers)
    for p, s in zip(papers, rank_by_similarity(query_vector(query, interests), vecs)):
        p.similarity = float(s)
    return vecs


def score_preference(papers: list[Paper]) -> None:
    """Sets p.preference when a preference model is trained. Call it after the LLM has ranked
    these papers: the model also reads relevance, recruiter score, datasets and GPU needs.
    Embeddings come from the cache the similarity step filled."""
    probs = predict(paper_vectors(papers), [p.to_dict() for p in papers])
    if probs is not None:
        for p, prob in zip(papers, probs):
            p.preference = float(prob)
