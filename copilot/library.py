"""Your library: every paper you've saved, rated or summarized, stored in library.json."""
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import Paper

try:
    import fcntl
except ImportError:  # Windows: the thread lock alone, so one server process at a time there
    fcntl = None  # type: ignore[assignment]

PATH = Path(__file__).resolve().parent.parent / "library.json"
_lock = threading.Lock()


@contextmanager
def _locked() -> Iterator[None]:
    """Every read-modify-write of library.json holds this: the thread lock for concurrent requests,
    plus an OS file lock so a second server process (uvicorn --workers, a script) can't interleave
    and overwrite another's save."""
    with _lock:
        lock_file = PATH.with_suffix(".lock")
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "a") as fh:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl:
                    fcntl.flock(fh, fcntl.LOCK_UN)


def _load() -> dict:
    data: dict = json.loads(PATH.read_text()) if PATH.exists() else {}
    return data


def _save(data: dict) -> None:
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(PATH)


def entries() -> list[dict]:
    with _locked():
        return sorted(_load().values(), key=lambda e: e["updated"], reverse=True)


def get(key: str) -> dict | None:
    with _locked():
        return _load().get(key)


def _entry(data: dict, paper: Paper, now: float) -> dict:
    entry: dict = data.get(paper.key) or {
        "key": paper.key, "rating": 0, "saved": False, "summary": "", "full_text": False, "added": now,
    }
    entry.setdefault("folders", [])
    entry["paper"] = paper.to_dict()
    return entry


def upsert(paper: Paper, **changes: object) -> dict:
    with _locked():
        data = _load()
        now = time.time()
        entry = _entry(data, paper, now)
        entry.update(changes, updated=now)
        data[paper.key] = entry
        _save(data)
        return entry


def set_folder(paper: Paper, folder: str, add: bool) -> dict:
    """Put a paper in a folder (which also saves it) or take it out. A paper can be in several;
    a folder exists while it holds at least one paper."""
    folder = " ".join(folder.split())[:60]
    if not folder:
        raise ValueError("folder name is empty")
    with _locked():
        data = _load()
        now = time.time()
        entry = _entry(data, paper, now)
        folders = [f for f in entry["folders"] if f != folder]
        if add:
            folders.append(folder)
            entry["saved"] = True
        entry.update(folders=sorted(folders, key=str.lower), updated=now)
        data[paper.key] = entry
        _save(data)
        return entry


def folders() -> list[dict]:
    """Every folder with how many papers it holds, alphabetically."""
    counts: dict[str, int] = {}
    for e in entries():
        for f in e.get("folders", []):
            counts[f] = counts.get(f, 0) + 1
    return [{"name": f, "count": counts[f]} for f in sorted(counts, key=str.lower)]


def remove(key: str) -> None:
    with _locked():
        data = _load()
        if data.pop(key, None) is not None:
            _save(data)


def rated_titles() -> tuple[list[str], list[str]]:
    """(liked, disliked) titles, oldest first so the ranker's [-15:] keeps the most recent."""
    es = sorted(entries(), key=lambda e: e["updated"])
    liked = [e["paper"]["title"] for e in es if e["rating"] > 0]
    disliked = [e["paper"]["title"] for e in es if e["rating"] < 0]
    return liked, disliked
