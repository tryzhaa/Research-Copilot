"""Ranking metrics over binary relevance (1 = you liked it, 0 = you didn't).

Each takes `ranked_labels`: the labels of one search's candidates in the order a strategy
put them. Only labeled candidates are passed in ("condensed list" evaluation), since most
candidates in a pool are never rated.
"""
import math
from collections.abc import Sequence


def precision_at_k(ranked_labels: Sequence[int], k: int) -> float:
    """Fraction of the top k that you liked. Divides by k even when fewer than k are judged,
    so every strategy is measured against the same bar."""
    return sum(ranked_labels[:k]) / k


def dcg_at_k(ranked_labels: Sequence[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(ranked_labels[:k]))


def ndcg_at_k(ranked_labels: Sequence[int], k: int) -> float:
    """DCG normalized by the best possible order of the same labels. 1.0 = every like ranked
    above every dislike. Undefined (returned as 0) when there are no likes."""
    ideal = dcg_at_k(sorted(ranked_labels, reverse=True), k)
    return dcg_at_k(ranked_labels, k) / ideal if ideal else 0.0


def reciprocal_rank(ranked_labels: Sequence[int]) -> float:
    """1 / position of the first like; 0 if there is none. Averaged over searches → MRR."""
    return next((1 / (i + 1) for i, rel in enumerate(ranked_labels) if rel), 0.0)


def evaluate(ranked_labels: Sequence[int]) -> dict[str, float]:
    return {
        "p@5": precision_at_k(ranked_labels, 5),
        "p@10": precision_at_k(ranked_labels, 10),
        "ndcg@10": ndcg_at_k(ranked_labels, 10),
        "mrr": reciprocal_rank(ranked_labels),
    }
