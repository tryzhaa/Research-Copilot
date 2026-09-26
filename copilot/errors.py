"""Typed failures. Each one carries enough context to show the user what broke and why.

A search degrades rather than fails: a broken source drops out, a broken or slow
ranker falls back to embedding similarity. These errors are how that gets reported.
"""
import httpx


class CopilotError(Exception):
    """Base class. `error_type` is a short machine-readable reason the UI can group on."""

    source: str = "copilot"
    error_type: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "error_type": self.error_type, "message": str(self)}


class SourceFetchError(CopilotError):
    """A paper source (arXiv, OpenAlex, ...) failed. The search continues without it."""

    def __init__(self, source: str, field: str, cause: Exception):
        self.source, self.field, self.cause = source, field, cause
        self.error_type = classify(cause)
        super().__init__(f"{source} ({field}): {_describe(cause)}")

    def to_dict(self) -> dict[str, str]:
        return super().to_dict() | {"field": self.field}


class RankingError(CopilotError):
    source = "ranker"
    error_type = "ranking_failed"


class RankingTimeoutError(RankingError):
    error_type = "timeout"


class RewriteError(CopilotError):
    """Query rewriting failed or was slow. The search goes ahead with the query as typed."""
    source = "rewriter"
    error_type = "rewrite_failed"


class EmbeddingError(CopilotError):
    source = "embeddings"
    error_type = "embedding_failed"


class SourceTimeout(Exception):
    """A source hadn't answered by the search's deadline, so the search went on without it."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        super().__init__(f"took over {seconds:g} s")


def classify(e: Exception) -> str:
    if isinstance(e, (httpx.TimeoutException, SourceTimeout)):
        return "timeout"
    if isinstance(e, httpx.HTTPStatusError):
        return "rate_limit" if e.response.status_code == 429 else "http_error"
    if isinstance(e, httpx.TransportError):
        return "network"
    return "bad_response"  # the source answered, but not with what we could parse


def _describe(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 429:
            return "rate limited, try again in a minute (a free API key avoids this, see preferences.yaml)"
        if code in (401, 403):
            return "rejected the API key, check it in .env"
        return f"HTTP {code}"
    if isinstance(e, SourceTimeout):
        return f"took over {e.seconds:g} s, skipped (source_timeout_seconds in preferences.yaml)"
    if isinstance(e, httpx.TimeoutException):
        return "timed out"
    if isinstance(e, httpx.TransportError):
        return "couldn't connect"
    return f"unexpected response ({type(e).__name__}: {e})"
