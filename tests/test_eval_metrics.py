"""Hand-computed examples, so the numbers in the eval table are verified, not just plausible."""
import math

import pytest

from eval.metrics import dcg_at_k, evaluate, ndcg_at_k, precision_at_k, reciprocal_rank


def test_precision_at_k() -> None:
    assert precision_at_k([1, 0, 1, 1, 0, 0], 5) == pytest.approx(3 / 5)
    assert precision_at_k([1, 1], 5) == pytest.approx(2 / 5)  # divides by k, not by list length
    assert precision_at_k([], 5) == 0


def test_dcg_by_hand() -> None:
    # 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5
    assert dcg_at_k([1, 0, 1], 10) == pytest.approx(1.5)


def test_ndcg_by_hand() -> None:
    # actual [0, 1, 1]: 1/log2(3) + 1/log2(4); ideal [1, 1, 0]: 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([0, 1, 1], 10) == pytest.approx(expected)
    assert expected == pytest.approx(0.6934, abs=1e-4)


def test_ndcg_bounds() -> None:
    assert ndcg_at_k([1, 1, 0, 0], 10) == 1.0
    assert ndcg_at_k([0, 0, 0], 10) == 0.0


def test_ndcg_cutoff_ignores_items_below_k() -> None:
    assert ndcg_at_k([0, 0, 1], 2) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([0, 0, 1, 1]) == pytest.approx(1 / 3)
    assert reciprocal_rank([1]) == 1.0
    assert reciprocal_rank([0, 0]) == 0.0


def test_evaluate_keys() -> None:
    assert set(evaluate([1, 0])) == {"p@5", "p@10", "ndcg@10", "mrr"}
