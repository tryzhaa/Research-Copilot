"""Model calls: rank candidates against your interests, and summarize one paper in your template.

Two providers, picked by `provider` in preferences.yaml:
  ollama    — free, open-weight models running locally (default)
  anthropic — Claude via the API (paid)
"""
import base64
import io
import os

import httpx
from pydantic import BaseModel

from .models import Paper

MAX_PDF_BYTES = 25 * 1024 * 1024
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
RANK_BATCH = 8  # small local models rank more reliably, and fit their context, a few papers at a time

RANK_SYSTEM = (
    "You are a research assistant ranking papers for one specific researcher. "
    "Score each paper 0-10 for how well it matches their current query AND standing interests. "
    "Reward rigor and genuine novelty; penalize papers that only match on keywords. "
    "Give a one-sentence reason that names what specifically makes it relevant or not."
)
SUMMARY_SYSTEM = (
    "You write precise research summaries for one researcher. Follow their template exactly. "
    "Only state what the paper supports; when you have the full text, cite the section or page "
    "for key claims, e.g. (§3.2) or (p. 5). If you only have the abstract, say so and do not "
    "invent details beyond it. Use LaTeX ($...$) for math."
)


class Score(BaseModel):
    index: int
    score: float  # 0-10
    reason: str


class Ranking(BaseModel):
    scores: list[Score]


# ---------- shared ----------

def _rank_prompt(papers: list[Paper], query: str, prefs: dict, liked: list[str], disliked: list[str], abstract_chars: int) -> str:
    listing = "\n\n".join(
        f"[{i}] {p.title} ({p.year}, {p.venue or p.source}, {p.citations} citations)\n{p.abstract[:abstract_chars]}"
        for i, p in enumerate(papers)
    )
    feedback = ""
    if liked:
        feedback += "\nPapers I rated highly before:\n" + "\n".join(f"- {t}" for t in liked[-15:])
    if disliked:
        feedback += "\nPapers I rated poorly before:\n" + "\n".join(f"- {t}" for t in disliked[-15:])
    return (f"My standing interests:\n{prefs['interests']}{feedback}\n\n"
            f"Current query: {query}\n\nCandidates:\n{listing}\n\nScore every candidate.")


def _apply_scores(papers: list[Paper], ranking: Ranking) -> None:
    for s in ranking.scores:
        if 0 <= s.index < len(papers):
            papers[s.index].score = max(0.0, min(10.0, s.score))
            papers[s.index].reason = s.reason


def _summary_request(paper: Paper, prefs: dict, template: str, has_full_text: bool) -> str:
    meta = f"Title: {paper.title}\nAuthors: {', '.join(paper.authors)}\nYear: {paper.year}\nVenue: {paper.venue}\n"
    if not has_full_text:
        meta += f"\nAbstract (full text unavailable):\n{paper.abstract}\n"
    return (f"{meta}\nMy interests:\n{prefs['interests']}\n\n"
            f"Summarize this paper using exactly this template:\n\n{template}")


def _fetch_pdf(url: str) -> bytes | None:
    try:
        r = httpx.get(url, follow_redirects=True, timeout=60, headers={"User-Agent": "research-copilot/0.1"})
        if r.status_code == 200 and r.content[:4] == b"%PDF" and len(r.content) < MAX_PDF_BYTES:
            return r.content
    except httpx.HTTPError:
        pass
    return None


def rank(papers: list[Paper], query: str, prefs: dict, liked: list[str], disliked: list[str]) -> list[Paper]:
    if not papers:
        return []
    if prefs.get("provider", "ollama") == "anthropic":
        _rank_claude(papers, query, prefs, liked, disliked)
    else:
        for start in range(0, len(papers), RANK_BATCH):
            _rank_ollama(papers[start:start + RANK_BATCH], query, prefs, liked, disliked)
    return sorted(papers, key=lambda p: p.score if p.score is not None else -1, reverse=True)


def summarize(paper: Paper, prefs: dict, template: str) -> tuple[str, bool]:
    """Returns (markdown summary, used_full_text)."""
    pdf = _fetch_pdf(paper.pdf_url) if paper.pdf_url else None
    if prefs.get("provider", "ollama") == "anthropic":
        return _summarize_claude(paper, prefs, template, pdf)
    return _summarize_ollama(paper, prefs, template, pdf)


# ---------- ollama (local, open-weight) ----------

def _ollama_chat(prefs: dict, system: str, user: str, schema: dict | None = None) -> str:
    body = {
        "model": prefs["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": prefs.get("effort") == "high",
        "options": {"num_ctx": prefs.get("context_tokens", 12288), "temperature": 0.2},
    }
    if schema:
        body["format"] = schema
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=900)
    except httpx.ConnectError:
        raise RuntimeError("Can't reach Ollama. Start it with `ollama serve` (or open the Ollama app).")
    if r.status_code == 404:
        raise RuntimeError(f"Model {prefs['model']} isn't installed. Run `ollama pull {prefs['model']}`.")
    r.raise_for_status()
    return r.json()["message"]["content"]


def _rank_ollama(papers: list[Paper], query: str, prefs: dict, liked: list[str], disliked: list[str]) -> None:
    prompt = _rank_prompt(papers, query, prefs, liked, disliked, abstract_chars=700)
    raw = _ollama_chat(prefs, RANK_SYSTEM, prompt, schema=Ranking.model_json_schema())
    _apply_scores(papers, Ranking.model_validate_json(raw))


def _pdf_text(pdf: bytes, max_chars: int) -> str:
    """Plain text with page markers so the model can cite pages. Truncated to fit the context window."""
    from pypdf import PdfReader

    out, total = [], 0
    for n, page in enumerate(PdfReader(io.BytesIO(pdf)).pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        chunk = f"\n[p. {n}]\n{text}"
        if total + len(chunk) > max_chars:
            out.append(chunk[:max_chars - total] + "\n[... truncated to fit the model's context ...]")
            break
        out.append(chunk)
        total += len(chunk)
    return "".join(out).strip()


def _summarize_ollama(paper: Paper, prefs: dict, template: str, pdf: bytes | None) -> tuple[str, bool]:
    # ~4 chars per token; leave room for the template, instructions and the answer.
    budget = max(4000, (prefs.get("context_tokens", 12288) - 4000) * 4)
    text = ""
    if pdf:
        try:
            text = _pdf_text(pdf, budget)
        except Exception:
            text = ""
    has_full_text = len(text) > 1500  # scanned PDFs extract to almost nothing
    prompt = _summary_request(paper, prefs, template, has_full_text)
    if has_full_text:
        prompt = f"Full text of the paper:\n<paper>\n{text}\n</paper>\n\n{prompt}"
    return _ollama_chat(prefs, SUMMARY_SYSTEM, prompt).strip(), has_full_text


# ---------- anthropic (Claude API) ----------

# Server-side refusal fallback: if the model declines, the API re-runs on a fallback model.
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
_claude = None


def _client():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic()
    return _claude


def _rank_claude(papers: list[Paper], query: str, prefs: dict, liked: list[str], disliked: list[str]) -> None:
    response = _client().beta.messages.parse(
        model=prefs["model"],
        max_tokens=16000,
        output_config={"effort": prefs.get("effort", "medium")},
        system=RANK_SYSTEM,
        messages=[{"role": "user", "content": _rank_prompt(papers, query, prefs, liked, disliked, abstract_chars=1200)}],
        output_format=Ranking,
        **FALLBACK,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError("Ranking was declined or returned no result.")
    _apply_scores(papers, response.parsed_output)


def _summarize_claude(paper: Paper, prefs: dict, template: str, pdf: bytes | None) -> tuple[str, bool]:
    content: list[dict] = []
    if pdf:
        content.append({
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": base64.standard_b64encode(pdf).decode()},
        })
    content.append({"type": "text", "text": _summary_request(paper, prefs, template, bool(pdf))})

    with _client().beta.messages.stream(
        model=prefs["model"],
        max_tokens=64000,
        output_config={"effort": prefs.get("effort", "medium")},
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": content}],
        **FALLBACK,
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        return "_The model declined to summarize this paper._", bool(pdf)
    text = "".join(b.text for b in message.content if b.type == "text")
    return text, bool(pdf)
