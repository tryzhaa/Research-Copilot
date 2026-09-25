import json

import pytest
from pytest_httpx import HTTPXMock

from copilot import llm
from tests.conftest import paper

PREFS = {"provider": "groq", "model": "m", "interests": "topology", "context_tokens": 8192}
GROQ = "https://api.groq.com/openai/v1/chat/completions"


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def _reply(content: str, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def _scores(n: int) -> str:
    return json.dumps({"scores": [
        {"index": i, "relevance": 12 if i == 0 else i, "relevance_reason": "r", "recruiter": -1, "recruiter_reason": "q",
         "datasets": ["CIFAR-10 dataset", "molecular data", "cifar-10"], "needs_gpu": False, "compute_note": "cpu"}
        for i in range(n)]})


def test_rank_via_hosted_model_parses_clamps_and_sorts(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=GROQ, json=_reply("<think>hmm</think>" + _scores(3)))
    ps = [paper("a"), paper("b"), paper("c")]
    out = llm.rank(ps, "q", PREFS, ["liked one"], ["disliked one"])
    assert [p.title for p in out] == ["a", "c", "b"]
    assert out[0].score == 10.0 and out[0].recruiter == 0.0  # clamped to 0-10
    assert out[0].datasets == ["CIFAR-10"]  # "dataset" suffix stripped, descriptions and dupes dropped
    req = json.loads(httpx_mock.get_requests()[0].content)
    assert req["response_format"] == {"type": "json_object"}
    assert "liked one" in req["messages"][1]["content"] and "disliked one" in req["messages"][1]["content"]
    assert httpx_mock.get_requests()[0].headers["authorization"] == "Bearer test-key"


def test_short_rate_limit_is_waited_out_once(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=GROQ, status_code=429, headers={"retry-after": "5"})
    httpx_mock.add_response(url=GROQ, json=_reply("hello"))
    assert llm._openai_chat(PREFS, "s", "u", 10) == "hello"


def test_long_rate_limit_fails_with_advice(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=GROQ, status_code=429, headers={"retry-after": "90"})
    with pytest.raises(RuntimeError, match="rate limit hit"):
        llm._openai_chat(PREFS, "s", "u", 10)


@pytest.mark.parametrize("status, match", [(413, "too large"), (500, "returned 500")])
def test_http_errors_are_readable(httpx_mock: HTTPXMock, status: int, match: str) -> None:
    httpx_mock.add_response(url=GROQ, status_code=status, text="nope")
    with pytest.raises(RuntimeError, match=match):
        llm._openai_chat(PREFS, "s", "u", 10)


def test_truncated_json_is_an_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=GROQ, json=_reply('{"scores": [', finish="length"))
    with pytest.raises(RuntimeError, match="ran out of room"):
        llm._openai_chat(PREFS, "s", "u", 10, json_mode=True)


def test_missing_key_and_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(RuntimeError, match="Set GROQ_API_KEY"):
        llm._openai_chat(PREFS, "s", "u", 10)
    with pytest.raises(RuntimeError, match="Unknown provider"):
        llm._openai_chat(PREFS | {"provider": "nope"}, "s", "u", 10)


def test_summary_uses_abstract_when_there_is_no_pdf(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=GROQ, json=_reply("## TL;DR\nok"))
    text, full = llm.summarize(paper("t", abstract="the abstract"), PREFS, "## TL;DR")
    assert text.startswith("## TL;DR") and full is False
    prompt = json.loads(httpx_mock.get_requests()[0].content)["messages"][1]["content"]
    assert "Abstract (full text unavailable):\nthe abstract" in prompt


def test_rank_of_nothing_calls_nothing() -> None:
    assert llm.rank([], "q", PREFS, [], []) == []
