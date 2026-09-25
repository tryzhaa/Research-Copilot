"""Persistent embedding cache in data/embeddings.sqlite, keyed by (model, paper key).

A text hash is stored alongside each vector, so a paper whose abstract changed (e.g. a
later source filled in a missing one) is re-embedded instead of served stale.
"""
import hashlib
import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

from .embeddings import MODEL_NAME, embed_texts

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings.sqlite"
log = logging.getLogger("uvicorn.error")

_lock = threading.Lock()
stats = {"hits": 0, "misses": 0}


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS vecs (model TEXT, key TEXT, text_hash TEXT, vec BLOB, PRIMARY KEY (model, key))")
    return db


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def get_embeddings(keys: list[str], texts: list[str], path: Path = DB_PATH) -> np.ndarray:
    """Vectors for each (key, text), computing and storing only the ones not cached yet."""
    hashes = [_hash(t) for t in texts]
    with _lock:
        db = _connect(path)
        try:
            found = {}
            for i in range(0, len(keys), 500):  # SQLite caps bound parameters
                chunk = keys[i:i + 500]
                found.update({k: (h, v) for k, h, v in db.execute(
                    f"SELECT key, text_hash, vec FROM vecs WHERE model = ? AND key IN ({','.join('?' * len(chunk))})",
                    [MODEL_NAME, *chunk])})
            missing = [i for i, (k, h) in enumerate(zip(keys, hashes)) if found.get(k, ("",))[0] != h]
            new = embed_texts([texts[i] for i in missing])
            db.executemany("INSERT OR REPLACE INTO vecs VALUES (?, ?, ?, ?)",
                           [(MODEL_NAME, keys[i], hashes[i], new[j].tobytes()) for j, i in enumerate(missing)])
            db.commit()
        finally:
            db.close()

    out = np.zeros((len(keys), new.shape[1] if len(new) else 384), dtype=np.float32)
    fresh = dict(zip(missing, new))
    for i, k in enumerate(keys):
        out[i] = fresh[i] if i in fresh else np.frombuffer(found[k][1], dtype=np.float32)

    hits = len(keys) - len(missing)
    stats["hits"] += hits
    stats["misses"] += len(missing)
    total = stats["hits"] + stats["misses"]
    log.info("embeddings: %d/%d cached this search, %.0f%% hit rate overall", hits, len(keys),
             100 * stats["hits"] / max(1, total))
    return out
