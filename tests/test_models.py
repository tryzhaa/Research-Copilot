import pytest
from fastapi.testclient import TestClient

from app import app
from copilot.models import InvalidPaper, Paper


def test_from_dict_drops_keys_that_arent_fields() -> None:
    # What the browser sends back: a serialized paper plus UI state.
    p = Paper.from_dict({"title": "Deep Sets", "year": 2017, "key": "arxiv:1703.06114", "rating": 1, "bibtex": "@article{}"})
    assert p.title == "Deep Sets"
    assert "rating" not in p.to_dict()


def test_from_dict_coerces_json_numbers_and_strings() -> None:
    p = Paper.from_dict({"title": "T", "year": "2020", "citations": 3.0, "similarity": 1})
    assert (p.year, p.citations, p.similarity) == (2020, 3, 1.0)


@pytest.mark.parametrize("bad, field", [
    ({"abstract": "no title"}, "title"),
    ({"title": "T", "year": "soon"}, "year"),
    ({"title": "T", "authors": "Ada Lovelace"}, "authors"),  # a string, not a list
    ({"title": "T", "citations": 2.5}, "citations"),
])
def test_from_dict_rejects_malformed_papers(bad: dict, field: str) -> None:
    with pytest.raises(InvalidPaper, match=field):
        Paper.from_dict(bad)


def test_round_trip_through_dict_is_lossless() -> None:
    p = Paper(title="T", authors=["A B"], datasets=["QM9", "ZINC"], needs_gpu=False, score=7.5)
    assert Paper.from_dict(p.to_dict()) == p


def test_api_answers_422_for_an_invalid_paper() -> None:
    r = TestClient(app).post("/api/rate", json={"paper": {"title": "T", "year": "soon"}, "rating": 1})
    assert r.status_code == 422
    assert "year" in r.json()["detail"]
