from dataclasses import dataclass, field, asdict
import re


@dataclass
class Paper:
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    pdf_url: str = ""
    citations: int = 0
    source: str = ""
    field: str = ""
    # Filled in by the ranker
    score: float | None = None
    reason: str = ""

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

    def to_dict(self) -> dict:
        return asdict(self)

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
