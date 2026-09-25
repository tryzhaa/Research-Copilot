import pytest

from copilot.models import Paper


@pytest.fixture
def prefs() -> dict:
    return {
        "candidates_per_source": 5,
        "filters": {"min_year": 2015, "min_citations": 0, "exclude_keywords": ["survey"]},
        "fields": {"ml": {"label": "ML", "arxiv_categories": ["cs.LG"], "openalex_field": 17, "s2_field": "Computer Science"}},
        "priorities": {"code_first": True, "min_datasets": 2, "prefer_cpu": True},
        "interests": "generative models and topology",
    }


def paper(title: str, **kw: object) -> Paper:
    return Paper(title=title, **kw)  # type: ignore[arg-type]
