"""Papers with Code, via its archive on Hugging Face.

paperswithcode.com shut down in July 2025 (it now redirects to huggingface.co/papers).
Its paper→repository links survive as a dataset, pwc-archive/links-between-paper-and-code,
and its dataset catalogue (15k benchmarks with their homepages) as pwc-archive/datasets.
We read the columns we need once, keep them in a small local SQLite index
(data/pwc_code.sqlite), and look papers up by arXiv ID and datasets up by name. Newer papers get their code
links from the live Hugging Face Papers source instead (see sources.search_hf_papers).

Build it with:  python -m copilot.pwc
"""
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

LINKS_PARQUET = ("https://huggingface.co/datasets/pwc-archive/links-between-paper-and-code"
                 "/resolve/main/data/train-00000-of-00001.parquet")
DATASETS_PARQUET = "https://huggingface.co/api/datasets/pwc-archive/datasets/parquet/default/train/0.parquet"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pwc_code.sqlite"

_building = threading.Lock()


def status() -> str:
    if DB_PATH.exists():
        return "ready"
    building_elsewhere = DB_PATH.with_suffix(".building").exists()  # e.g. `python -m copilot.pwc` running
    return "building" if _building.locked() or building_elsewhere else "missing"


def build() -> int:
    """Download the links table (columns only) and write the local index. Returns row count."""
    import duckdb

    DB_PATH.parent.mkdir(exist_ok=True)
    tmp = DB_PATH.with_suffix(".building")
    with _building:
        tmp.unlink(missing_ok=True)
        tmp.touch()  # marks "building" for other processes while the download runs
        try:
            con = duckdb.connect()
            con.execute("INSTALL httpfs; LOAD httpfs;")
            rows = con.execute(f"""
                SELECT regexp_replace(paper_arxiv_id, 'v[0-9]+$', '') AS arxiv_id,
                       repo_url, is_official, coalesce(framework, 'none')
                FROM read_parquet('{LINKS_PARQUET}')
                WHERE paper_arxiv_id IS NOT NULL AND paper_arxiv_id <> '' AND repo_url IS NOT NULL
            """).fetchall()
            db = sqlite3.connect(tmp)
            db.execute("CREATE TABLE code (arxiv_id TEXT, repo_url TEXT, official INTEGER, framework TEXT)")
            db.executemany("INSERT INTO code VALUES (?, ?, ?, ?)", rows)
            db.execute("CREATE INDEX idx_arxiv ON code (arxiv_id)")
            db.commit()
            db.close()
            build_datasets(tmp, con)
            tmp.replace(DB_PATH)
            return len(rows)
        finally:
            tmp.unlink(missing_ok=True)


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_datasets(path: Path = DB_PATH, con: Any = None) -> int:
    """Add the dataset catalogue to the index: normalized name → name and homepage. Official names
    win over variants ("ZINC 100k" → ZINC), and among clashes the most-used dataset wins."""
    import duckdb
    if con is None:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
    rows = con.execute(f"""
        SELECT name, coalesce(homepage, ''), coalesce(variants, []), coalesce(num_papers, 0)
        FROM read_parquet('{DATASETS_PARQUET}') WHERE name IS NOT NULL ORDER BY 4 DESC
    """).fetchall()
    by_key: dict[str, tuple[str, str]] = {}
    for name, homepage, _, _ in rows:
        by_key.setdefault(norm_name(name), (name, homepage))
    for name, homepage, variants, _ in rows:
        for v in variants:
            by_key.setdefault(norm_name(v), (name, homepage))
    by_key.pop("", None)
    db = sqlite3.connect(path)
    try:
        db.execute("DROP TABLE IF EXISTS datasets")
        db.execute("CREATE TABLE datasets (key TEXT PRIMARY KEY, name TEXT, homepage TEXT)")
        db.executemany("INSERT INTO datasets VALUES (?, ?, ?)", [(k, n, h) for k, (n, h) in by_key.items()])
        db.commit()
    finally:
        db.close()
    return len(by_key)


def _has_datasets() -> bool:
    db = sqlite3.connect(DB_PATH)
    try:
        return db.execute("SELECT 1 FROM sqlite_master WHERE name = 'datasets'").fetchone() is not None
    finally:
        db.close()


def build_in_background() -> None:
    if status() == "missing":
        threading.Thread(target=build, daemon=True).start()
    elif status() == "ready" and not _has_datasets():  # an index built before datasets were added
        threading.Thread(target=build_datasets, daemon=True).start()


def search_link(name: str) -> str:
    return f"https://huggingface.co/datasets?search={quote(name)}"


def dataset_links(names: list[str]) -> dict[str, str]:
    """name → where to find the dataset: its homepage from the Papers with Code catalogue, else a
    Hugging Face dataset search. Size suffixes fall back to the base dataset (ZINC250K → ZINC)."""
    if not names:
        return {}
    found: dict[str, str] = {}
    if DB_PATH.exists():
        db = sqlite3.connect(DB_PATH)
        try:
            for name in names:
                key = norm_name(name)
                for k in dict.fromkeys([key, re.sub(r"(\d+k|\d+m|full|small|large|v\d+)$", "", key)]):
                    try:
                        row = db.execute("SELECT homepage FROM datasets WHERE key = ?", (k,)).fetchone()
                    except sqlite3.OperationalError:  # index built before datasets were added
                        row = None
                    if row and row[0].startswith("http"):
                        found[name] = row[0]
                        break
        finally:
            db.close()
    return {n: found.get(n) or search_link(n) for n in names}


def lookup(arxiv_ids: list[str]) -> dict[str, dict]:
    """arxiv_id → best repo: official first, then one whose framework is known."""
    ids = sorted({re.sub(r"v\d+$", "", i) for i in arxiv_ids if i})
    if not ids or not DB_PATH.exists():
        return {}
    db = sqlite3.connect(DB_PATH)
    try:
        rows = db.execute(
            f"SELECT arxiv_id, repo_url, official, framework FROM code WHERE arxiv_id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
    finally:
        db.close()
    best: dict[str, dict] = {}
    for arxiv_id, repo, official, framework in rows:
        rank = (official, framework != "none")
        cur = best.get(arxiv_id)
        if cur is None or rank > cur["_rank"]:
            best[arxiv_id] = {"repo": repo, "official": bool(official), "framework": framework, "_rank": rank}
    return best


if __name__ == "__main__":
    print(f"Building {DB_PATH} from the Papers with Code archive (one-time, ~30 MB download)...")
    print(f"Indexed {build():,} paper→code links and the dataset catalogue.")
