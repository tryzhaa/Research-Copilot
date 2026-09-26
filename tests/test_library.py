from pathlib import Path

import pytest

from copilot import library, snapshots
from tests.conftest import paper


@pytest.fixture(autouse=True)
def tmp_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "PATH", tmp_path / "library.json")


def test_rate_save_and_remove() -> None:
    a, b = paper("A", doi="10.1/a"), paper("B", doi="10.1/b")
    library.upsert(a, rating=1)
    library.upsert(b, rating=-1, saved=True)
    library.upsert(a, summary="s")  # keeps the rating
    assert library.get(a.key)["rating"] == 1 and library.get(a.key)["summary"] == "s"
    assert library.rated_titles() == (["A"], ["B"])
    library.remove(b.key)
    assert library.get(b.key) is None and len(library.entries()) == 1


def test_snapshot_round_trip_records_what_each_stage_saw(tmp_path: Path) -> None:
    pool = [paper("A", doi="10.1/a", similarity=0.8), paper("B", doi="10.1/b", similarity=0.4)]
    ranked = [paper("A", doi="10.1/a", similarity=0.8, score=7.0)]
    snapshots.save("sheaf diffusion", {"interests": "i", "model": "m"}, ["ml"], pool, ranked, ["Liked title"],
                   rewrite={"keywords": "k", "intent": "i"}, directory=tmp_path)
    [s] = snapshots.load_all(tmp_path)
    assert s["query"] == "sheaf diffusion" and s["feedback_titles"] == ["Liked title"]
    assert s["rewrite"] == {"keywords": "k", "intent": "i"}
    a, b = s["candidates"]
    assert a["shortlisted"] and a["score"] == 7.0
    assert not b["shortlisted"] and b["score"] is None and b["similarity"] == 0.4


def test_folders_hold_papers_and_saving_to_one_saves_the_paper() -> None:
    a, b = paper("A", doi="10.1/a"), paper("B", doi="10.1/b")
    e = library.set_folder(a, "  Topology   ideas ", add=True)
    assert e["folders"] == ["Topology ideas"] and e["saved"]
    library.set_folder(a, "art", add=True)
    library.set_folder(a, "art", add=True)  # adding twice doesn't duplicate
    library.set_folder(b, "art", add=True)
    assert library.get(a.key)["folders"] == ["art", "Topology ideas"]
    assert library.folders() == [{"name": "art", "count": 2}, {"name": "Topology ideas", "count": 1}]
    library.set_folder(a, "Topology ideas", add=False)
    assert library.folders() == [{"name": "art", "count": 2}]  # an emptied folder disappears
    library.upsert(a, rating=1)
    assert library.get(a.key)["folders"] == ["art"]  # other updates keep folders


def test_folder_name_must_not_be_blank() -> None:
    with pytest.raises(ValueError):
        library.set_folder(paper("A"), "   ", add=True)


def test_save_endpoint_saves_and_unsaving_empties_folders(monkeypatch: pytest.MonkeyPatch) -> None:
    # The page has one save button: it saves, the picker files, "unsave" undoes both.
    from fastapi.testclient import TestClient

    from app import app
    monkeypatch.delenv("DEMO_MODE", raising=False)
    client = TestClient(app)
    p = {"title": "Deep Sets", "doi": "10.1/a"}
    assert client.post("/api/save", json={"paper": p, "saved": True}).json() == {"saved": True, "folders": [], "all": []}
    client.post("/api/folder", json={"paper": p, "folder": "art", "add": True})
    body = client.post("/api/save", json={"paper": p, "saved": False}).json()
    assert body == {"saved": False, "folders": [], "all": []}  # out of "art", which is now empty and gone
    assert library.get("doi:10.1/a")["folders"] == []
