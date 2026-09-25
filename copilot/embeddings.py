"""Sentence embeddings for papers and queries.

fastembed runs BAAI/bge-small-en-v1.5 (384 dims) through ONNX Runtime on the CPU. We use
it instead of sentence-transformers because it doesn't pull in PyTorch: a ~130 MB model
download instead of ~2 GB of dependencies, and it embeds ~60 abstracts in well under a
second on a laptop. The tradeoff is no training API — see scripts/ for what that rules out.
"""
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .errors import EmbeddingError

if TYPE_CHECKING:
    from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Attention memory grows with batch x 512^2 tokens: fastembed's default batch (256) put ~45
# abstracts through at once and peaked near 1 GB, twice what small hosts allow. 8 costs a
# little speed and keeps the whole app well under 512 MB.
BATCH_SIZE = 8
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

_model: "TextEmbedding | None" = None


def get_model() -> "TextEmbedding":
    global _model
    if _model is None:
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(MODEL_NAME, cache_dir=str(MODEL_DIR), threads=cpu_limit())
        except Exception as e:
            raise EmbeddingError(f"couldn't load {MODEL_NAME}: {e}") from e
    return _model


def cpu_limit(cpu_max: Path = Path("/sys/fs/cgroup/cpu.max")) -> int | None:
    """CPUs a container may actually use (its cgroup quota), or None outside one.

    ONNX Runtime otherwise starts a thread per host core: in a container limited to 1 CPU on an
    8-core machine, embedding 45 abstracts took 53.6 s with 8 threads and 7.6 s with 1."""
    try:
        quota, period = cpu_max.read_text().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return None


def normalize(vecs: np.ndarray) -> np.ndarray:
    """Unit-length rows (or vector), so a dot product is cosine similarity."""
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    out: np.ndarray = vecs / np.maximum(norms, 1e-12)
    return out


def embed_texts(texts: list[str]) -> np.ndarray:
    """Documents (paper title + abstract) → (n, 384) float32, unit length."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    try:
        return normalize(np.array(list(get_model().embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32))
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"embedding failed: {e}") from e


def embed_query(text: str) -> np.ndarray:
    """A search query → (384,) unit vector. bge prefixes queries with a retrieval instruction."""
    try:
        return normalize(np.array(list(get_model().query_embed([text]))[0], dtype=np.float32))
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"embedding failed: {e}") from e


def paper_text(title: str, abstract: str) -> str:
    return f"{title}. {abstract}".strip()
