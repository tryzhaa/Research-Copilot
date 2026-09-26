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

# Concurrency: the lock guards only the quick database reads and writes. Computing embeddings
# (the slow part, ~0.3 s a paper on one CPU) happens outside it, limited to COMPUTE_SLOTS at a
# time: on a 1-CPU host, parallel runs aren't faster and each adds ~130 MB, so one visitor's search
# no longer blocks another's cached lookups, and several can't run the memory out together.
COMPUTE_SLOTS = 1
_lock = threading.Lock()
_compute = threading.BoundedSemaphore(COMPUTE_SLOTS)
_stats_lock = threading.Lock()
stats = {"hits": 0, "misses": 0}


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)  # other processes (a second worker, a script) may be writing
    db.execute("PRAGMA journal_mode=WAL")    # readers don't wait for writers
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
        finally:
            db.close()
    missing = [i for i, (k, h) in enumerate(zip(keys, hashes)) if found.get(k, ("",))[0] != h]
    new = np.zeros((0, 384), dtype=np.float32)
    if missing:
        with _compute:
            new = embed_texts([texts[i] for i in missing])
        with _lock:
            db = _connect(path)
            try:
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
    with _stats_lock:
        stats["hits"] += hits
        stats["misses"] += len(missing)
        total = stats["hits"] + stats["misses"]
    log.info("embeddings: %d/%d cached this search, %.0f%% hit rate overall", hits, len(keys),
             100 * stats["hits"] / max(1, total))
    return out
