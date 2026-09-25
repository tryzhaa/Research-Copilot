from pathlib import Path

import numpy as np
import pytest

from copilot import embed_cache, embeddings


@pytest.fixture
def fake_embed(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Deterministic 'embedding': [len(text), 1, 0, ...]. Records every batch it's asked to embed."""
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, :2] = [len(t), 1]
        return out

    monkeypatch.setattr(embed_cache, "embed_texts", embed)
    return calls


def test_cache_only_embeds_what_it_hasnt_seen(tmp_path: Path, fake_embed: list[list[str]]) -> None:
    db = tmp_path / "e.sqlite"
    a = embed_cache.get_embeddings(["k1", "k2"], ["aa", "bbb"], db)
    b = embed_cache.get_embeddings(["k2", "k3", "k1"], ["bbb", "c", "aa"], db)
    assert fake_embed == [["aa", "bbb"], ["c"]]
    assert np.array_equal(a[0], b[2]) and np.array_equal(a[1], b[0])
    assert b[1, 0] == 1  # "c"


def test_changed_text_is_reembedded(tmp_path: Path, fake_embed: list[list[str]]) -> None:
    db = tmp_path / "e.sqlite"
    embed_cache.get_embeddings(["k1"], ["no abstract"], db)
    v = embed_cache.get_embeddings(["k1"], ["now with a longer abstract"], db)
    assert fake_embed[-1] == ["now with a longer abstract"]
    assert v[0, 0] == len("now with a longer abstract")


def test_all_cached_embeds_nothing(tmp_path: Path, fake_embed: list[list[str]]) -> None:
    db = tmp_path / "e.sqlite"
    embed_cache.get_embeddings(["k1"], ["aa"], db)
    embed_cache.get_embeddings(["k1"], ["aa"], db)
    assert fake_embed[1] == []


def test_cpu_limit_reads_the_cgroup_quota(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    for content, expected in [("50000 100000\n", 1), ("150000 100000\n", 2), ("max 100000\n", None)]:
        cpu_max.write_text(content)
        assert embeddings.cpu_limit(cpu_max) == expected
    assert embeddings.cpu_limit(tmp_path / "missing") is None
