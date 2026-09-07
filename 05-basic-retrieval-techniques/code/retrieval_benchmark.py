"""
Module 5 — Retrieval Benchmark: Similarity vs MMR vs Hybrid (RRF)
=================================================================
Head-to-head on a FIXED small corpus (data/samples/* -> chunks) and 5 sample
questions. Three retrieval methods, all implemented with numpy + local
MiniLM embeddings (no LLM, no vector store):

    A. PLAIN SIMILARITY   top-3 by cosine similarity
    B. MMR                maximal marginal relevance (λ=0.5, fetch_k=10)
    C. HYBRID RRF         dense top-10 ⊕ BM25 top-10 fused with RRF (k=60)

Metrics per method:
    • keyword hit-rate of top-3  — did any expected keyword land in top-3?
    • diversity                  — 1 − mean pairwise cosine within top-3
    • latency (ms)               — median over repeated runs

Run from the project root:
    uv run python 05-basic-retrieval-techniques/code/retrieval_benchmark.py
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib
import re
import time
from pathlib import Path

# Make `shared.config` importable no matter which directory we launch from.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_embeddings

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# 1. FIXED CORPUS + FIXED QUESTIONS (deterministic benchmark)
# ---------------------------------------------------------------------------
def load_chunks() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        for i, piece in enumerate(splitter.split_text(text)):
            chunks.append(Document(
                page_content=piece,
                metadata={"source": path.name, "id": f"{path.name}::chunk{i}"}))
    return chunks


# (question, keywords that should appear in a good top-3 answer)
QUESTIONS: list[tuple[str, list[str]]] = [
    ("How does RAG reduce hallucination?", ["ground", "retriev", "context"]),
    ("What are HNSW parameters like M and efSearch?", ["hnsw", "efsearch", "graph"]),
    ("Which vector database should I pick?", ["vector", "database", "chroma"]),
    ("What embedding model is used locally?", ["minilm", "embedding", "sentence"]),
    ("What is reciprocal rank fusion?", ["reciprocal", "rank", "fusion"]),
]


# ---------------------------------------------------------------------------
# 2. THE THREE METHODS (each returns top-3 chunk indices for one query)
# ---------------------------------------------------------------------------
def retrieve_similarity(cos: np.ndarray, k: int = 3) -> list[int]:
    """A. Plain semantic top-k."""
    return [int(i) for i in np.argsort(-cos)[:k]]


def retrieve_mmr(chunk_vecs: np.ndarray, cos: np.ndarray, k: int = 3,
                 fetch_k: int = 10, lam: float = 0.5) -> list[int]:
    """B. Maximal Marginal Relevance — greedy: relevance vs redundancy."""
    pool = set(int(i) for i in np.argsort(-cos)[:fetch_k])
    selected: list[int] = []
    while len(selected) < k and pool:
        best, best_score = None, -np.inf
        for i in pool:
            if selected:
                red = max(chunk_vecs[i] @ chunk_vecs[j] for j in selected)
            else:
                red = 0.0
            score = lam * cos[i] - (1 - lam) * red
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        pool.discard(best)
    return selected


def rrf_fuse(list_a: list[int], list_b: list[int], n_docs: int, k: int = 60) -> np.ndarray:
    """C. Reciprocal Rank Fusion of two ranked index lists (pure numpy)."""
    scores = np.zeros(n_docs, dtype=np.float64)
    for ranked in (list_a, list_b):
        inv = 1.0 / (k + np.arange(1, len(ranked) + 1))
        np.add.at(scores, np.asarray(ranked, dtype=np.int64), inv)
    return scores


def retrieve_hybrid(chunk_vecs: np.ndarray, q_vec: np.ndarray, bm25: BM25Okapi,
                    query: str, n_docs: int, k: int = 3, fetch: int = 10) -> list[int]:
    """C. Dense top-10 ⊕ BM25 top-10, fused with RRF, take top-k."""
    cos = chunk_vecs @ q_vec
    dense_top = [int(i) for i in np.argsort(-cos)[:fetch]]
    b_scores = bm25.get_scores(tokenize(query))
    sparse_top = [int(i) for i in np.argsort(-b_scores)[:fetch]]
    fused = rrf_fuse(dense_top, sparse_top, n_docs)
    return [int(i) for i in np.argsort(-fused)[:k]]


# ---------------------------------------------------------------------------
# 3. METRICS
# ---------------------------------------------------------------------------
def keyword_hit(top_chunks: list[str], keywords: list[str]) -> bool:
    """True if ANY expected keyword appears in ANY top-3 chunk (substring)."""
    blob = " ".join(top_chunks).lower()
    return any(kw in blob for kw in keywords)


def diversity(chunk_vecs: np.ndarray, idxs: list[int]) -> float:
    """1 − mean pairwise cosine within the returned set (higher = less redundant)."""
    if len(idxs) < 2:
        return 1.0
    vecs = chunk_vecs[idxs]
    sim = vecs @ vecs.T
    iu = np.triu_indices(len(idxs), k=1)
    return float(1.0 - sim[iu].mean())


def measure_latency(fn, reps: int = 7) -> float:
    """Median wall-clock ms over `reps` runs (first run warms caches)."""
    fn()  # warm-up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    console.rule("[bold]MODULE 5 · RETRIEVAL BENCHMARK · SIMILARITY vs MMR vs HYBRID")

    chunks = load_chunks()
    n = len(chunks)
    console.print(f"[green]Corpus: {n} chunks · {len(QUESTIONS)} fixed questions.[/green]")

    texts = [c.page_content for c in chunks]
    emb = get_embeddings()
    chunk_vecs = np.array(emb.embed_documents(texts), dtype=np.float32)
    norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    chunk_vecs = chunk_vecs / np.where(norms == 0, 1.0, norms)
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    methods = {}
    for q, _ in QUESTIONS:
        q_vec = np.asarray(emb.embed_query(q), dtype=np.float32)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-9)
        cos = chunk_vecs @ q_vec
        methods[q] = {
            "similarity": lambda c=cos: retrieve_similarity(c),
            "mmr": lambda cv=chunk_vecs, c=cos: retrieve_mmr(cv, c),
            "hybrid_rrf": lambda cv=chunk_vecs, qv=q_vec, qq=q:
                retrieve_hybrid(cv, qv, bm25, qq, n),
        }

    # ---- collect metrics ----
    rows = []
    for name, fn_key in [("Plain Similarity", "similarity"),
                         ("MMR (λ=0.5)", "mmr"),
                         ("Hybrid BM25⊕Dense (RRF)", "hybrid_rrf")]:
        hits, divs, lats = 0, [], []
        for q, kws in QUESTIONS:
            fns = methods[q][fn_key]
            top = fns()
            hits += keyword_hit([chunks[i].page_content for i in top], kws)
            divs.append(diversity(chunk_vecs, top))
            lats.append(measure_latency(fns))
        rows.append((name, hits / len(QUESTIONS), float(np.mean(divs)),
                     float(np.median(lats))))

    table = Table(title="Head-to-head on the fixed corpus",
                  title_style="bold cyan", header_style="bold magenta")
    table.add_column("method", style="bold")
    table.add_column("keyword hit-rate (top-3)", justify="right")
    table.add_column("diversity (1−mean cos)", justify="right")
    table.add_column("latency (ms, median)", justify="right")
    for name, hr, div, lat in rows:
        bar = "█" * round(hr * 10) + "░" * round((1 - hr) * 10)
        table.add_row(name, f"[green]{bar}[/green] {hr:.0%}", f"{div:.3f}", f"{lat:.1f}")
    console.print(table)

    # ---- verdict ----
    best_hr = max(rows, key=lambda r: r[1])
    best_div = max(rows, key=lambda r: r[2])
    best_lat = min(rows, key=lambda r: r[3])
    console.print(Panel(
        f"[bold]VERDICT[/bold]\n"
        f"• Best keyword coverage : [cyan]{best_hr[0]}[/cyan] ({best_hr[1]:.0%} of questions)\n"
        f"• Most diverse top-3    : [cyan]{best_div[0]}[/cyan] (diversity {best_div[2]:.3f})\n"
        f"• Fastest               : [cyan]{best_lat[0]}[/cyan] ({best_lat[3]:.1f} ms)\n\n"
        "[dim]Reading the trade-off: plain similarity is cheapest but returns near-duplicate "
        "paraphrases (low diversity). MMR spends compute de-duplicating, which raises "
        "diversity at a small relevance cost. Hybrid RRF adds the lexical branch, so exact "
        "terms (HNSW, efSearch, MiniLM) that embeddings blur still land in top-3 — usually "
        "the best hit-rate at moderate latency.[/dim]",
        border_style="green"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as exc:  # noqa: BLE001 - friendly top-level handler
        console.print(Panel(f"[red]{type(exc).__name__}: {exc}[/red]", title="Startup error"))
        console.print("[dim]Make sure dependencies are installed (`uv sync`) and that the local "
                      "embedding model (all-MiniLM-L6-v2) can be downloaded on first run.[/dim]")
        sys.exit(0)
