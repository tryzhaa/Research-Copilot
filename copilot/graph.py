"""A similarity graph over every paper you've come across (saved searches + your library).

Each paper links to its K nearest neighbours by embedding cosine similarity (above a floor),
made symmetric. Two uses:

- similar(): papers related to one paper, by personalized PageRank from it: a random walk that
  keeps restarting at the paper's own neighbourhood. Direct neighbours score high, and so do
  papers two hops away that several of them point to, which plain nearest-neighbour search misses.
  Each result says which paper links it to the seed ("via").
- neighbourhood(): the nodes and edges around a set of papers, for drawing the map.

Built from cached embeddings in well under a second for a few thousand papers, and rebuilt only
when a new search is saved or the library changes.
"""
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from . import library, snapshots
from .embed_cache import get_embeddings
from .embeddings import embed_texts, paper_text

K = 8               # neighbours per paper
MIN_SIM = 0.6       # below this cosine similarity, bge-small pairs are rarely on the same topic
RESTART = 0.4       # PageRank restart probability: higher keeps results closer to the seed
ITERATIONS = 30

_lock = threading.Lock()
_cache: "tuple[tuple, Graph] | None" = None


@dataclass
class Graph:
    keys: list[str]
    papers: list[dict]
    vecs: np.ndarray            # (n, d) unit rows
    adj: sparse.csr_matrix      # (n, n) symmetric, weights = cosine similarity
    index: dict[str, int]


def paper_url(key: str, paper: dict) -> str:
    """Older snapshots didn't store URLs; rebuild one from the paper's ID."""
    if paper.get("url"):
        return str(paper["url"])
    kind, _, ident = key.partition(":")
    if kind == "arxiv":
        return f"https://arxiv.org/abs/{ident}"
    if kind == "doi":
        return f"https://doi.org/{ident}"
    return "https://scholar.google.com/scholar?q=" + str(paper.get("title", "")).replace(" ", "+")


def _collect(directory: Path = snapshots.DIR) -> dict[str, dict]:
    """key → paper metadata. Later searches overwrite earlier ones; the library wins over both."""
    papers: dict[str, dict] = {}
    for s in snapshots.load_all(directory):
        for c in s["candidates"]:
            papers[c["key"]] = {k: c.get(k) for k in ("title", "abstract", "year", "url", "code_url", "source")}
    for e in library.entries():
        p = e["paper"]
        papers[e["key"]] = {k: p.get(k) for k in ("title", "abstract", "year", "url", "code_url", "source")}
    for key, p in papers.items():
        p["url"] = paper_url(key, p)
    return papers


def knn_adjacency(vecs: np.ndarray, k: int = K, min_sim: float = MIN_SIM) -> sparse.csr_matrix:
    """Symmetric k-nearest-neighbour graph. Computed in row blocks so memory stays O(n·block)."""
    n = len(vecs)
    rows, cols, vals = [], [], []
    for start in range(0, n, 1024):
        sims = vecs[start:start + 1024] @ vecs.T
        for i, row in enumerate(sims):
            row[start + i] = -1.0  # no self-loops
            top = np.argpartition(-row, min(k, n - 1) - 1)[:k] if n > 1 else np.array([], dtype=int)
            for j in top:
                if row[j] >= min_sim:
                    rows.append(start + i); cols.append(int(j)); vals.append(float(row[j]))
    a = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return a.maximum(a.T).tocsr()  # an edge if either paper counts the other among its neighbours


def build(papers: dict[str, dict]) -> Graph:
    keys = list(papers)
    vecs = get_embeddings(keys, [paper_text(papers[k]["title"] or "", papers[k]["abstract"] or "") for k in keys])
    return Graph(keys, [papers[k] | {"key": k} for k in keys], vecs, knn_adjacency(vecs), {k: i for i, k in enumerate(keys)})


def current(directory: Path = snapshots.DIR) -> Graph:
    """The graph, rebuilt only when a search was saved or the library changed since last time."""
    global _cache
    sig = (len(list(directory.glob("*.json"))), library.PATH.stat().st_mtime if library.PATH.exists() else 0)
    with _lock:
        if _cache is None or _cache[0] != sig:
            _cache = (sig, build(_collect(directory)))
        return _cache[1]


def personalized_pagerank(adj: sparse.csr_matrix, seed: np.ndarray, restart: float = RESTART,
                          iterations: int = ITERATIONS) -> np.ndarray:
    """Stationary distribution of a walk that follows edges (by weight) and jumps back to `seed`
    with probability `restart` at each step."""
    deg = np.asarray(adj.sum(axis=1)).ravel()
    walk = sparse.diags(np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)) @ adj  # row-stochastic
    seed = seed / seed.sum()
    r = seed.copy()
    for _ in range(iterations):
        r = restart * seed + (1 - restart) * (walk.T @ r)
    return r


def similar(paper: dict, key: str, n_direct: int = 5, n_graph: int = 5, g: Graph | None = None) -> list[dict]:
    """Papers related to `paper`: its closest neighbours, then papers the walk reaches that aren't
    among them ("found through the graph"), each with the neighbour that links it. Works for
    papers not in the graph too."""
    g = g or current()
    if len(g.keys) < 2:
        return []
    if key in g.index:
        v = g.vecs[g.index[key]]
    else:
        v = embed_texts([paper_text(paper.get("title", ""), paper.get("abstract", ""))])[0]
    sims = g.vecs @ v
    self_i = g.index.get(key)
    if self_i is not None:
        sims[self_i] = -1.0

    # Restart at the paper's own nearest neighbours, weighted by how similar they are.
    direct = [int(i) for i in np.argsort(-sims)[:K] if sims[i] >= MIN_SIM]
    if not direct:
        return []
    seed = np.zeros(len(g.keys))
    seed[direct] = sims[direct]
    rank = personalized_pagerank(g.adj, seed)
    if self_i is not None:
        rank[self_i] = 0.0

    order = [int(i) for i in np.argsort(-rank) if rank[i] > 0]
    near = [i for i in order if i in direct][:n_direct]
    # A walk can drift along a chain of weak links into another topic; keep only papers that are
    # still similar to the seed in their own right.
    reached = [i for i in order if i not in direct and sims[i] >= MIN_SIM][:n_graph]

    def via(i: int) -> str | None:
        """The seed neighbour that links to i most strongly (weighted by its own similarity)."""
        links = [(g.adj[m, i] * sims[m], m) for m in direct if g.adj[m, i] > 0]
        return g.papers[max(links)[1]]["title"] if links else None

    return [g.papers[i] | {"similarity": round(float(sims[i]), 3), "score": float(rank[i]),
                           "direct": i in near, "via": None if i in near else via(i)}
            for i in near + reached]


def neighbourhood(focus: list[str], limit: int = 250, g: Graph | None = None) -> dict:
    """Focus papers plus their neighbours (strongest links first) and the edges among them."""
    g = g or current()
    chosen = [g.index[k] for k in dict.fromkeys(focus) if k in g.index][:limit]
    picked = set(chosen)
    frontier: list[tuple[float, int]] = []
    for i in chosen:
        row = g.adj.getrow(i)
        frontier += [(float(w), int(j)) for j, w in zip(row.indices, row.data) if j not in picked]
    for _, j in sorted(frontier, reverse=True):
        if len(picked) >= limit:
            break
        picked.add(j)
    nodes = sorted(picked)
    sub = g.adj[nodes][:, nodes].tocoo()
    focus_set = set(focus)
    return {
        "nodes": [{"key": g.keys[i], "title": g.papers[i]["title"], "year": g.papers[i]["year"],
                   "url": g.papers[i]["url"], "focus": g.keys[i] in focus_set} for i in nodes],
        "links": [{"source": g.keys[nodes[a]], "target": g.keys[nodes[b]], "sim": round(float(w), 3)}
                  for a, b, w in zip(sub.row, sub.col, sub.data) if a < b],
        "total_papers": len(g.keys),
    }



def related(focus: list[str], exclude: set[str], n: int = 12, g: Graph | None = None) -> list[dict]:
    """The n papers outside `focus` most similar to any focus paper, skipping `exclude` (e.g. ones
    you've rated). Each is the stored paper plus "sim" (its strongest link into the focus set) and
    "via" (the title of the focus paper it's closest to)."""
    g = g or current()
    focus_idx = [g.index[k] for k in dict.fromkeys(focus) if k in g.index]
    best: dict[int, tuple[float, int]] = {}
    for i in focus_idx:
        row = g.adj.getrow(i)
        for j, w in zip(row.indices, row.data):
            key = g.keys[j]
            if key in exclude or key in focus:
                continue
            if w > best.get(int(j), (0.0, -1))[0]:
                best[int(j)] = (float(w), i)
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:n]
    return [g.papers[j] | {"sim": round(w, 3), "via": g.papers[i]["title"]} for j, (w, i) in top]
