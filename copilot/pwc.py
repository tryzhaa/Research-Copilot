"""Papers with Code, via its archive on Hugging Face.

paperswithcode.com shut down in July 2025 (it now redirects to huggingface.co/papers).
Its paper→repository links survive as a dataset, pwc-archive/links-between-paper-and-code.
We read the three columns we need once, keep them in a small local SQLite index
(data/pwc_code.sqlite), and look papers up by arXiv ID. Newer papers get their code
links from the live Hugging Face Papers source instead (see sources.search_hf_papers).

Build it with:  python -m copilot.pwc
"""
import re
import sqlite3
import threading
from pathlib import Path

LINKS_PARQUET = ("https://huggingface.co/datasets/pwc-archive/links-between-paper-and-code"
                 "/resolve/main/data/train-00000-of-00001.parquet")
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
            tmp.replace(DB_PATH)
            return len(rows)
        finally:
            tmp.unlink(missing_ok=True)


def build_in_background() -> None:
    if status() == "missing":
        threading.Thread(target=build, daemon=True).start()


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
    print(f"Indexed {build():,} paper→code links.")
