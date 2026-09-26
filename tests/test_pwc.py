import sqlite3
from pathlib import Path

import pytest

from copilot import pwc, search
from tests.conftest import paper


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "pwc.sqlite"
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE code (arxiv_id TEXT, repo_url TEXT, official INTEGER, framework TEXT)")
    db.executemany("INSERT INTO code VALUES (?, ?, ?, ?)", [
        ("2202.04579", "https://github.com/fan/reimpl", 0, "pytorch"),
        ("2202.04579", "https://github.com/twitter/sheaf", 1, "pytorch"),
        ("1706.03762", "https://github.com/a/unknown", 0, "none"),
        ("1706.03762", "https://github.com/b/tf", 0, "tf"),
    ])
    db.commit()
    db.close()
    monkeypatch.setattr(pwc, "DB_PATH", db_path)
    return db_path


def test_lookup_prefers_official_then_known_framework(index: Path) -> None:
    found = pwc.lookup(["2202.04579v2", "1706.03762", "9999.99999", ""])
    assert found["2202.04579"]["repo"] == "https://github.com/twitter/sheaf"
    assert found["2202.04579"]["official"] is True
    assert found["1706.03762"]["repo"] == "https://github.com/b/tf"
    assert "9999.99999" not in found


def test_lookup_without_index_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pwc, "DB_PATH", tmp_path / "missing.sqlite")
    assert pwc.lookup(["2202.04579"]) == {}
    assert pwc.status() == "missing"


def test_attach_code_keeps_a_live_repo_unless_the_index_has_the_official_one(index: Path) -> None:
    live = paper("A", arxiv_id="1706.03762", code_url="https://github.com/live/repo")
    official = paper("B", arxiv_id="2202.04579", code_url="https://github.com/live/other")
    bare = paper("C", arxiv_id="1706.03762")
    search.attach_code([live, official, bare])
    assert live.code_url == "https://github.com/live/repo"
    assert official.code_url == "https://github.com/twitter/sheaf" and official.code_official
    assert bare.code_url == "https://github.com/b/tf" and bare.code_framework == "tf"


def test_datasets_link_to_their_homepage_else_a_hugging_face_search(index: Path) -> None:
    db = sqlite3.connect(index)
    db.execute("CREATE TABLE datasets (key TEXT PRIMARY KEY, name TEXT, homepage TEXT)")
    db.executemany("INSERT INTO datasets VALUES (?, ?, ?)", [
        ("qm9", "QM9", "http://quantum-machine.org/datasets/"),
        ("zinc", "ZINC", "http://zinc15.docking.org/"),
        ("cifar10", "CIFAR-10", ""),
    ])
    db.commit()
    db.close()
    links = pwc.dataset_links(["QM9", "ZINC250K", "CIFAR-10", "Made Up Set"])
    assert links["QM9"] == "http://quantum-machine.org/datasets/"
    assert links["ZINC250K"] == "http://zinc15.docking.org/"          # a size variant finds its base
    assert links["CIFAR-10"] == "https://huggingface.co/datasets?search=CIFAR-10"  # no homepage
    assert links["Made Up Set"] == "https://huggingface.co/datasets?search=Made%20Up%20Set"


def test_an_index_built_before_datasets_still_gives_search_links(index: Path) -> None:
    assert pwc.dataset_links(["QM9"]) == {"QM9": "https://huggingface.co/datasets?search=QM9"}
    assert pwc.dataset_links([]) == {}
