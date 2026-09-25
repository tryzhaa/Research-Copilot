import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Paper(BaseModel):
    """One paper as it moves through the pipeline. Validated wherever it enters: sources,
    the browser (every /api route that takes a paper), library.json and the eval dataset."""

    # Unknown keys are dropped, so a paper can be rebuilt from any dict that carries its fields
    # (API payloads also hold key, rating, bibtex, ...). Assignments aren't re-validated: the
    # pipeline fills scores in place, and those values come from typed code or validated LLM output.
    model_config = ConfigDict(extra="ignore")

    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    pdf_url: str = ""
    citations: int = 0
    source: str = ""
    field: str = ""
    # Code, from Papers with Code / Hugging Face
    code_url: str = ""
    code_official: bool = False
    code_framework: str = ""
    stars: int = 0
    # Filled in by retrieval (cosine similarity to query + interests, -1..1) and the preference model (P(like), 0..1)
    similarity: float | None = None
    preference: float | None = None
    # Filled in by the ranker
    score: float | None = None          # relevance to query + interests, 0-10
    reason: str = ""
    recruiter: float | None = None      # how impressive an implementation would look to recruiters, 0-10
    recruiter_reason: str = ""
    datasets: list[str] = Field(default_factory=list)  # datasets it's evaluated on
    needs_gpu: bool | None = None       # to replicate the core result
    compute_note: str = ""

    @property
    def has_code(self) -> bool:
        return bool(self.code_url)

    @property
    def ids(self) -> list[str]:
        """Every identifier this record can be matched on. Two records sharing any one are the same paper."""
        ids = []
        doi = self.doi.lower()
        arxiv = re.sub(r"v\d+$", "", self.arxiv_id)
        if doi.startswith("10.48550/arxiv."):  # arXiv's own DOI, as OpenAlex reports preprints
            arxiv = arxiv or doi.removeprefix("10.48550/arxiv.")
        elif doi:
            ids.append("doi:" + doi)
        if arxiv:
            ids.append("arxiv:" + arxiv.lower())
        ids.append("title:" + re.sub(r"\W+", "", self.title.lower())[:80])
        return ids

    @property
    def key(self) -> str:
        """Stable ID used for the library."""
        return self.ids[0]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        """Build from untrusted input. Raises InvalidPaper with a readable message."""
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'paper'}: {err['msg']}" for err in e.errors())
            raise InvalidPaper(problems) from None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def bibtex(self) -> str:
        first = (self.authors[0].split()[-1] if self.authors else "anon").lower()
        cite_key = re.sub(r"\W", "", f"{first}{self.year or ''}{self.title.split()[0] if self.title else ''}").lower()
        lines = [f"@article{{{cite_key},", f"  title = {{{self.title}}},"]
        if self.authors:
            lines.append(f"  author = {{{' and '.join(self.authors)}}},")
        if self.year:
            lines.append(f"  year = {{{self.year}}},")
        if self.venue:
            lines.append(f"  journal = {{{self.venue}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.arxiv_id:
            lines.append(f"  eprint = {{{self.arxiv_id}}}, archivePrefix = {{arXiv}},")
        lines.append(f"  url = {{{self.url}}}\n}}")
        return "\n".join(lines)


class InvalidPaper(ValueError):
    """A paper dict that doesn't fit the Paper schema, e.g. year="soon" from a hand-edited library."""
