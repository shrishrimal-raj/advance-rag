"""
Module 5 — Basic Retrieval Techniques (Retrievers)
==================================================
Builds ONE shared corpus from data/samples/*, chunks it, embeds it into an
IN-MEMORY ChromaDB collection, then runs the SAME three test queries through
FIVE different retrievers and prints a side-by-side comparison:

  1. Similarity Search            (top-k, semantic)
  2. Similarity Score Threshold   (min_score -> anti-hallucination gate)
  3. Maximal Marginal Relevance   (MMR: relevance vs diversity)
  4. BM25 Sparse Retrieval        (keyword / term-frequency)
  5. Hybrid Search                (dense + sparse fused with Reciprocal Rank Fusion)
     + an Ensemble-style weighted merge for comparison

No LLM is required for this module — only local embeddings.

Run from the project root:
    uv run python 05-basic-retrieval-techniques/code/main.py
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib
import re
from pathlib import Path

# Make `shared.config` importable no matter which directory we launch from.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_embeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

# ---------------------------------------------------------------------------
# 1. BUILD THE SHARED CORPUS (load -> chunk -> embed into in-memory Chroma)
# ---------------------------------------------------------------------------
def load_raw_documents() -> list[Document]:
    """Read every sample file into a Document tagged with source/format."""
    docs = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": path.name,
                      "format": path.suffix.lstrip(".").lower()},
        ))
    return docs


def build_chunks(raw_docs: list[Document]) -> list[Document]:
    """Chunk every raw document and give each chunk a stable id."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in raw_docs:
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            meta = dict(doc.metadata)
            meta["id"] = f"{doc.metadata['source']}::chunk{i}"
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Embed chunks into an IN-MEMORY Chroma collection (cosine space)."""
    console.print("[dim]Embedding chunks with local sentence-transformers…[/dim]")
    return Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name="m5_corpus",
        collection_metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine_sim
    )


# ---------------------------------------------------------------------------
# 2. SPARSE (BM25) INDEX + RECIPROCAL RANK FUSION helpers
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — plenty for BM25 on English text."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25(chunks: list[Document]) -> BM25Okapi:
    """Fit a BM25 index over the tokenized chunks."""
    corpus_tokens = [tokenize(c.page_content) for c in chunks]
    return BM25Okapi(corpus_tokens)


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion across several ranked lists.

    score(id) = sum over lists of 1 / (k + rank), rank is 1-based.
    k=60 dampens the impact of any single list's top position.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# 3. PRETTY PRINTING
# ---------------------------------------------------------------------------
def snippet(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def print_results(title: str, rows: list) -> None:
    """rows: (rank, score_or_None, source, snippet)."""
    if not rows:
        console.print(Panel("[yellow]∅ NO RESULTS — nothing cleared the bar[/yellow]",
                            title=title, border_style="yellow"))
        return
    table = Table(title=title, title_style="bold cyan", header_style="bold magenta")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("score", justify="right", width=9)
    table.add_column("source", width=22)
    table.add_column("snippet", overflow="fold")
    for rank, score, source, snip in rows:
        table.add_row(str(rank),
                      f"{score:.4f}" if isinstance(score, float) else "—",
                      source, snip)
    console.print(table)


# ---------------------------------------------------------------------------
# 4. THE FIVE RETRIEVERS (each returns printable rows for one query)
# ---------------------------------------------------------------------------
def r_similarity(vs: Chroma, query: str, k: int = 3) -> list:
    """(1) Plain semantic top-k. Chroma returns cosine distance -> flip to similarity."""
    rows = []
    for i, (doc, dist) in enumerate(vs.similarity_search_with_score(query, k=k), 1):
        rows.append((i, 1.0 - dist, doc.metadata.get("source", "?"),
                     snippet(doc.page_content)))
    return rows


def r_threshold(vs: Chroma, query: str, min_score: float = 0.45,
                k: int = 3, n: int = 1) -> list:
    """(2) Anti-hallucination gate: keep only chunks with similarity >= min_score.

    We scan the WHOLE collection so a truly off-topic query can return EMPTY.
    """
    kept = []
    for doc, dist in vs.similarity_search_with_score(query, k=n):
        sim = 1.0 - dist
        if sim >= min_score:
            kept.append((doc, sim))
    kept.sort(key=lambda x: x[1], reverse=True)
    return [(i, sim, doc.metadata.get("source", "?"), snippet(doc.page_content))
            for i, (doc, sim) in enumerate(kept[:k], 1)]


def r_mmr(vs: Chroma, query: str, k: int = 3, fetch_k: int = 20,
          lambda_mult: float = 0.5) -> list:
    """(3) Maximal Marginal Relevance: balance relevance against redundancy."""
    docs = vs.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    return [(i, None, d.metadata.get("source", "?"), snippet(d.page_content))
            for i, d in enumerate(docs, 1)]


def r_bm25(bm25: BM25Okapi, chunks: list[Document], query: str, k: int = 3) -> list:
    """(4) Sparse keyword retrieval via BM25 (term frequency x inverse doc freq)."""
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(rank, float(scores[idx]), chunks[idx].metadata.get("source", "?"),
             snippet(chunks[idx].page_content))
            for rank, idx in enumerate(order, 1)]


def r_hybrid(dense_top: list, sparse_top: list, chunks: list[Document],
             k_rrf: int = 60, k: int = 3) -> list:
    """(5) HYBRID: fuse dense + sparse rankings with Reciprocal Rank Fusion."""
    dense_ids = [doc_id for doc_id, _ in dense_top]
    sparse_ids = [doc_id for doc_id, _ in sparse_top]
    fused = rrf_fuse([dense_ids, sparse_ids], k=k_rrf)
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    by_id = {c.metadata["id"]: c for c in chunks}
    return [(rank, score, by_id[doc_id].metadata.get("source", "?"),
             snippet(by_id[doc_id].page_content))
            for rank, (doc_id, score) in enumerate(ranked, 1)]


def r_ensemble(dense_top: list, sparse_top: list, chunks: list[Document],
               w_dense: float = 0.5, w_sparse: float = 0.5, k: int = 3) -> list:
    """(5b) ENSEMBLE: weighted linear merge of min-max-normalized branch scores."""
    def minmax(pairs):
        vals = [v for _, v in pairs]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return {doc_id: (v - lo) / rng for doc_id, v in pairs}
    dn, sn = minmax(dense_top), minmax(sparse_top)
    merged = {i: w_dense * dn.get(i, 0.0) + w_sparse * sn.get(i, 0.0)
              for i in set(dn) | set(sn)}
    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:k]
    by_id = {c.metadata["id"]: c for c in chunks}
    return [(rank, score, by_id[doc_id].metadata.get("source", "?"),
             snippet(by_id[doc_id].page_content))
            for rank, (doc_id, score) in enumerate(ranked, 1)]


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    console.rule("[bold]MODULE 5 · BASIC RETRIEVAL TECHNIQUES")

    raw = load_raw_documents()
    console.print(f"[green]Loaded {len(raw)} source files from data/samples/.[/green]")
    chunks = build_chunks(raw)
    console.print(f"[green]Created {len(chunks)} chunks.[/green]")
    vs = build_vectorstore(chunks)
    bm25 = build_bm25(chunks)
    n = len(chunks)

    # Three queries: two on-topic, one deliberately off-topic (to trip the gate).
    queries = [
        "How does RAG reduce hallucination?",
        "What are HNSW parameters like M and efSearch?",
        "How do I bake chocolate chip cookies?",
    ]

    for q in queries:
        console.print(Panel(f"[bold white]QUERY:[/bold white] {q}", border_style="blue"))

        print_results("1 · Similarity Search (top-3)", r_similarity(vs, q, k=3))
        print_results("2 · Similarity Score Threshold (min_score=0.45)",
                      r_threshold(vs, q, min_score=0.45, n=n))
        print_results("3 · Maximal Marginal Relevance (k=3, fetch_k=20, λ=0.5)",
                      r_mmr(vs, q))
        print_results("4 · BM25 Sparse (top-3)", r_bm25(bm25, chunks, q, k=3))

        # Prepare the two branches used by BOTH hybrid (RRF) and ensemble (weights).
        dense = vs.similarity_search_with_score(q, k=10)
        dense_top = [(d.metadata["id"], 1.0 - dist) for d, dist in dense]
        scores = bm25.get_scores(tokenize(q))
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:10]
        sparse_top = [(chunks[i].metadata["id"], float(scores[i])) for i in order]

        print_results("5 · HYBRID Search (dense ⊕ BM25, RRF k=60)",
                      r_hybrid(dense_top, sparse_top, chunks))
        print_results("5b · ENSEMBLE weighted merge (0.5·dense + 0.5·sparse)",
                      r_ensemble(dense_top, sparse_top, chunks))
        console.print()

    console.rule("[bold]DONE")
    console.print(
        "[dim]Compare the columns: the threshold gate drops the off-topic cookie query, "
        "MMR diversifies the sources, BM25 nails exact keywords, and hybrid/ensemble "
        "blend both signals for the most robust ranking.[/dim]"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as exc:  # noqa: BLE001 - friendly top-level handler
        console.print(Panel(f"[red]{type(exc).__name__}: {exc}[/red]", title="Startup error"))
        console.print("[dim]Make sure dependencies are installed (`uv sync`) and that the local "
                      "embedding model (all-MiniLM-L6-v2) can be downloaded on first run.[/dim]")
