"""Model calls: rank candidates against your interests, and summarize one paper in your template.

Providers, picked by `provider` in preferences.yaml:
  ollama    — free, open-weight models running locally
  anthropic — Claude via the API (paid)
  groq, cerebras, openrouter, gemini, together, openai — hosted models over the
              OpenAI-compatible chat API (see OPENAI_COMPATIBLE)
"""
import base64
import copy
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

from .errors import CopilotError, RankingError, RankingTimeoutError
from .models import Paper

if TYPE_CHECKING:
    import anthropic


def _load_dotenv() -> None:
    """API keys from a git-ignored .env (KEY=value per line). Real env vars win."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and not name.strip().startswith("#"):
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


_load_dotenv()
MAX_PDF_BYTES = 25 * 1024 * 1024
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
log = logging.getLogger("uvicorn.error")
RANK_BATCH = 8  # small local models rank more reliably, and fit their context, a few papers at a time

# provider → (base URL, env var holding the API key). `base_url` / `api_key_env` in prefs override these.
OPENAI_COMPATIBLE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}

RANK_SYSTEM = (
    "You are a research assistant helping one person pick papers to implement in code for their portfolio. "
    "Address them as \"you\" in reasons. "
    "For EVERY candidate, return:\n"
    "- relevance (0-10): fit with their current query AND standing interests. Reward rigor and genuine "
    "novelty; penalize papers that only match on keywords. relevance_reason: one sentence naming what "
    "specifically makes it relevant or not (max 20 words).\n"
    "- recruiter (0-10): how interesting and impressive a from-scratch implementation of this paper would "
    "look to a tech recruiter or hiring manager. High: a recognizable, current problem; a clear, "
    "demo-able result with measurable numbers; real engineering depth (not a thin wrapper); finishable "
    "by one person in a few weeks. Low: pure theory with nothing to build, trivial tweaks of a baseline, "
    "results that need a lab's compute, or problems nobody outside the subfield would recognize. "
    "recruiter_reason: one sentence on what would impress or what would fall flat (max 20 words).\n"
    "- datasets: named benchmark datasets the paper evaluates on, e.g. [\"CIFAR-10\", \"QM9\", \"ZINC\"]. "
    "Only proper names stated in the text. NOT descriptions like \"molecular data\" or \"3D structures\", "
    "and not chemical formulas. An empty list if none are named. Never guess.\n"
    "- needs_gpu: true if reproducing the core result realistically needs a GPU (training large "
    "networks, transformers beyond small scale, diffusion or LLM training, ImageNet-scale data). false "
    "if it runs on a laptop CPU in hours: classical ML, algorithms, small networks on small data, "
    "inference with small pretrained models, or theory. compute_note: at most 8 words, e.g. "
    "\"small MLP on MNIST, CPU fine\" or \"trains 1B-param model on 64 GPUs\"."
)
SUMMARY_SYSTEM = (
    "You write precise research summaries for one researcher. Follow their template exactly. "
    "Only state what the paper supports; when you have the full text, cite the section or page "
    "for key claims, e.g. (§3.2) or (p. 5). If you only have the abstract, say so and do not "
    "invent details beyond it. Use LaTeX ($...$) for math."
)


class Score(BaseModel):
    index: int
    relevance: float  # 0-10
    relevance_reason: str
    recruiter: float  # 0-10
    recruiter_reason: str
    datasets: list[str]
    needs_gpu: bool
    compute_note: str


class Ranking(BaseModel):
    scores: list[Score]


# ---------- shared ----------

def _rank_prompt(papers: list[Paper], query: str, prefs: dict, liked: list[str], disliked: list[str], abstract_chars: int) -> str:
    def code(p: Paper) -> str:
        if not p.has_code:
            return "no public code"
        bits = ["official code" if p.code_official else "public code"]
        if p.code_framework:
            bits.append(p.code_framework)
        if p.stars:
            bits.append(f"{p.stars} GitHub stars")
        return ", ".join(bits)

    listing = "\n\n".join(
        f"[{i}] {p.title} ({p.year}, {p.venue or p.source}, {p.citations} citations, {code(p)})\n"
        f"{p.abstract[:abstract_chars]}"
        for i, p in enumerate(papers)
    )
    feedback = ""
    if liked:
        feedback += "\nPapers I rated highly before:\n" + "\n".join(f"- {t}" for t in liked[-15:])
    if disliked:
        feedback += "\nPapers I rated poorly before:\n" + "\n".join(f"- {t}" for t in disliked[-15:])
    return (f"My standing interests:\n{prefs['interests']}{feedback}\n\n"
            f"Current query: {query}\n\nCandidates:\n{listing}\n\nAssess every candidate.")


def _apply_scores(papers: list[Paper], ranking: Ranking) -> None:
    for s in ranking.scores:
        if 0 <= s.index < len(papers):
            p = papers[s.index]
            p.score = max(0.0, min(10.0, s.relevance))
            p.reason = s.relevance_reason
            p.recruiter = max(0.0, min(10.0, s.recruiter))
            p.recruiter_reason = s.recruiter_reason
            p.datasets = _clean_datasets(s.datasets)
            p.needs_gpu = s.needs_gpu
            p.compute_note = s.compute_note


def _clean_datasets(names: list[str]) -> list[str]:
    """Keep proper dataset names; small models also return descriptions like "3D molecular data"."""
    out: list[str] = []
    for d in names:
        d = re.sub(r"\s+(dataset|datasets|benchmark)$", "", d.strip(), flags=re.I)
        if not d or re.search(r"\bdata\b", d, re.I) or not re.search(r"[A-Z0-9]", d):
            continue
        if d.lower() not in (o.lower() for o in out):
            out.append(d)
    return out[:12]


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
    provider = prefs.get("provider", "ollama")
    if provider == "anthropic":
        _rank_claude(papers, query, prefs, liked, disliked)
    elif provider != "ollama":
        _rank_openai(papers, query, prefs, liked, disliked)
    else:
        for start in range(0, len(papers), RANK_BATCH):
            _rank_ollama(papers[start:start + RANK_BATCH], query, prefs, liked, disliked)
    return sorted(papers, key=lambda p: p.score if p.score is not None else -1, reverse=True)


_rank_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rank")


def rank_with_timeout(papers: list[Paper], query: str, prefs: dict, liked: list[str], disliked: list[str],
                      timeout: float) -> list[Paper]:
    """rank(), but give up after `timeout` seconds so a slow or hung model can't stall the search.

    The model works on copies: if it times out and finishes later, it can't change papers
    the caller has already moved on with (and ranked by similarity instead).
    """
    fut = _rank_pool.submit(rank, copy.deepcopy(papers), query, prefs, liked, disliked)
    try:
        return fut.result(timeout=timeout)
    except FuturesTimeout:
        raise RankingTimeoutError(f"{prefs['model']} took over {timeout:.0f}s, so these are ranked by similarity. "
                                  "Raise rank_timeout_seconds in preferences.yaml for slow local models.") from None
    except CopilotError:
        raise
    except Exception as e:
        raise RankingError(f"ranking failed, so these are ranked by similarity: {e}") from e


def summarize(paper: Paper, prefs: dict, template: str) -> tuple[str, bool]:
    """Returns (markdown summary, used_full_text)."""
    pdf = _fetch_pdf(paper.pdf_url) if paper.pdf_url else None
    provider = prefs.get("provider", "ollama")
    if provider == "anthropic":
        return _summarize_claude(paper, prefs, template, pdf)
    if provider != "ollama":
        return _summarize_openai(paper, prefs, template, pdf)
    return _summarize_ollama(paper, prefs, template, pdf)


# ---------- ollama (local, open-weight) ----------

def _ollama_chat(prefs: dict, system: str, user: str, max_tokens: int, schema: dict | None = None) -> str:
    think = prefs.get("effort") == "high"
    body = {
        "model": prefs["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": think,
        "keep_alive": "30m",  # reloading a model costs ~30 s on an 8 GB Mac
        "options": {
            "num_ctx": prefs.get("context_tokens", 8192),
            "temperature": 0.2,
            # Hard cap. Small models can loop forever (especially reasoning models that
            # ignore think=false), and without a cap the request just hangs.
            "num_predict": max_tokens * (3 if think else 1),
        },
    }
    if schema:
        body["format"] = schema
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=httpx.Timeout(600, connect=5))
    except httpx.ConnectError:
        raise RuntimeError("Can't reach Ollama. Start it with `ollama serve` (or open the Ollama app).")
    except httpx.ReadTimeout:
        raise RuntimeError(f"{prefs['model']} took over 10 minutes. Try a smaller model or lower context_tokens.")
    if r.status_code == 404:
        raise RuntimeError(f"Model {prefs['model']} isn't installed. Run `ollama pull {prefs['model']}`.")
    r.raise_for_status()
    data = r.json()
    content = re.sub(r"<think>.*?(</think>|$)", "", data["message"].get("content", ""), flags=re.S).strip()
    if not content:
        why = "spent its whole budget thinking" if data["message"].get("thinking") else "returned nothing"
        raise RuntimeError(f"{prefs['model']} {why}. Try a non-reasoning model such as qwen3:4b.")
    if data.get("done_reason") == "length" and schema:
        raise RuntimeError(f"{prefs['model']} ran out of room before finishing its answer.")
    return content


def _rank_ollama(papers: list[Paper], query: str, prefs: dict, liked: list[str], disliked: list[str]) -> None:
    prompt = _rank_prompt(papers, query, prefs, liked, disliked, abstract_chars=700)
    t0 = time.monotonic()
    raw = _ollama_chat(prefs, RANK_SYSTEM, prompt, max_tokens=200 * len(papers) + 300, schema=Ranking.model_json_schema())
    log.info("ranked %d papers in %.0fs", len(papers), time.monotonic() - t0)
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


def _text_summary_prompt(paper: Paper, prefs: dict, template: str, pdf: bytes | None) -> tuple[str, bool]:
    """Summary prompt with the PDF as extracted text, for models that can't read PDFs directly."""
    # ~4 chars per token; leave room for the template, instructions and the answer.
    budget = max(4000, (prefs.get("context_tokens", 8192) - 4000) * 4)
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
    return prompt, has_full_text


def _summarize_ollama(paper: Paper, prefs: dict, template: str, pdf: bytes | None) -> tuple[str, bool]:
    prompt, has_full_text = _text_summary_prompt(paper, prefs, template, pdf)
    return _ollama_chat(prefs, SUMMARY_SYSTEM, prompt, max_tokens=3000), has_full_text


# ---------- hosted open models (OpenAI-compatible APIs: Groq, Cerebras, OpenRouter, ...) ----------

def _openai_chat(prefs: dict, system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    provider = prefs["provider"]
    base_url, key_env = OPENAI_COMPATIBLE.get(provider, (None, None))
    base_url, key_env = prefs.get("base_url", base_url), prefs.get("api_key_env", key_env)
    if not base_url or not key_env:
        raise RuntimeError(f"Unknown provider {provider!r}. Use ollama, anthropic, {', '.join(OPENAI_COMPATIBLE)}, "
                           "or set base_url and api_key_env in preferences.yaml.")
    key = os.getenv(key_env)
    if not key:
        raise RuntimeError(f"Set {key_env} to use {provider}: `export {key_env}=...` before starting the app.")
    body = {
        "model": prefs["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,  # reasoning models spend part of this thinking, so it's generous
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    for attempt in range(2):
        try:
            r = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body,
                           headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(180, connect=10))
        except httpx.TimeoutException:
            raise RuntimeError(f"{provider} didn't answer within 3 minutes.")
        wait = r.headers.get("retry-after")
        # Free tiers cap tokens per minute; back-to-back searches trip it. Wait once if it's short.
        if r.status_code == 429 and attempt == 0 and wait and float(wait) <= 30:
            log.info("%s rate limited, retrying in %ss", provider, wait)
            time.sleep(float(wait))
            continue
        break
    if r.status_code == 429:
        raise RuntimeError(f"{provider} rate limit hit{f' (retry in {wait}s)' if wait else ''}. "
                           "Wait a moment, or lower rank_at_most / context_tokens.")
    if r.status_code == 413:
        raise RuntimeError(f"Request too large for {provider}'s limits. Lower context_tokens in preferences.yaml.")
    if r.status_code >= 400:
        raise RuntimeError(f"{provider} returned {r.status_code}: {r.text[:300]}")
    choice = r.json()["choices"][0]
    content = re.sub(r"<think>.*?(</think>|$)", "", choice["message"].get("content") or "", flags=re.S).strip()
    if not content:
        raise RuntimeError(f"{prefs['model']} returned nothing.")
    if json_mode and choice.get("finish_reason") == "length":
        raise RuntimeError(f"{prefs['model']} ran out of room before finishing its answer.")
    return content


def _rank_openai(papers: list[Paper], query: str, prefs: dict, liked: list[str], disliked: list[str]) -> None:
    system = (f"{RANK_SYSTEM}\n\nReply with only a JSON object matching this JSON schema:\n"
              f"{json.dumps(Ranking.model_json_schema())}")
    prompt = _rank_prompt(papers, query, prefs, liked, disliked, abstract_chars=1200)
    t0 = time.monotonic()
    raw = _openai_chat(prefs, system, prompt, max_tokens=300 * len(papers) + 2000, json_mode=True)
    log.info("ranked %d papers in %.1fs", len(papers), time.monotonic() - t0)
    _apply_scores(papers, Ranking.model_validate_json(raw[raw.find("{"):raw.rfind("}") + 1]))


def _summarize_openai(paper: Paper, prefs: dict, template: str, pdf: bytes | None) -> tuple[str, bool]:
    prompt, has_full_text = _text_summary_prompt(paper, prefs, template, pdf)
    return _openai_chat(prefs, SUMMARY_SYSTEM, prompt, max_tokens=8000), has_full_text


# ---------- anthropic (Claude API) ----------

# Server-side refusal fallback: if the model declines, the API re-runs on a fallback model.
FALLBACK: dict[str, Any] = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
_claude: "anthropic.Anthropic | None" = None


def _client() -> "anthropic.Anthropic":
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
    content: list[Any] = []
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
