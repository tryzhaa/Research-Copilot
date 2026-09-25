"""Your library: every paper you've saved, rated or summarized, stored in library.json."""
import json
import threading
import time
from pathlib import Path

from .models import Paper

PATH = Path(__file__).resolve().parent.parent / "library.json"
_lock = threading.Lock()


def _load() -> dict:
    data: dict = json.loads(PATH.read_text()) if PATH.exists() else {}
    return data


def _save(data: dict) -> None:
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(PATH)


def entries() -> list[dict]:
    with _lock:
        return sorted(_load().values(), key=lambda e: e["updated"], reverse=True)


def get(key: str) -> dict | None:
    with _lock:
        return _load().get(key)


def upsert(paper: Paper, **changes: object) -> dict:
    with _lock:
        data = _load()
        now = time.time()
        entry = data.get(paper.key) or {
            "key": paper.key, "rating": 0, "saved": False, "summary": "", "full_text": False, "added": now,
        }
        entry["paper"] = paper.to_dict()
        entry.update(changes, updated=now)
        data[paper.key] = entry
        _save(data)
        return entry


def remove(key: str) -> None:
    with _lock:
        data = _load()
        if data.pop(key, None) is not None:
            _save(data)


def rated_titles() -> tuple[list[str], list[str]]:
    """(liked, disliked) titles, oldest first so the ranker's [-15:] keeps the most recent."""
    es = sorted(entries(), key=lambda e: e["updated"])
    liked = [e["paper"]["title"] for e in es if e["rating"] > 0]
    disliked = [e["paper"]["title"] for e in es if e["rating"] < 0]
    return liked, disliked
