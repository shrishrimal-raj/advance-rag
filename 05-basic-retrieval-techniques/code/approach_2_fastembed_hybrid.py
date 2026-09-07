"""
Module 5 — Approach B: Hybrid Retrieval with fastembed
======================================================
Same corpus as main.py (data/samples/* -> chunks), but the DENSE branch is
built with **fastembed** (Qdrant's ONNX-based embedding library) instead of
LangChain sentence-transformers:

    dense  : fastembed.TextEmbedding("sentence-transformers/all-MiniLM-L6-v2") -> cosine top-10
    sparse : rank_bm25.BM25Okapi                          -> BM25 top-10
    fusion : Reciprocal Rank Fusion (k=60)                -> fused top-5

Why fastembed? It runs ONNX models with zero GPU / zero LangChain overhead,
which is exactly how production search services embed at query time.

If fastembed cannot initialize (offline, download blocked, missing ONNX
runtime), we FALL BACK to the shared local sentence-transformers embeddings
(`shared.config.get_embeddings`) and say so — the script never crashes.

No LLM required. Run from the project root:
    uv run python 05-basic-retrieval-techniques/code/approach_2_fastembed_hybrid.py
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
# 1. CORPUS (identical recipe to main.py so results are comparable)
# ---------------------------------------------------------------------------
def load_raw_documents() -> list[Document]:
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


# ---------------------------------------------------------------------------
# 2. DENSE BRANCH — fastembed first, shared local embeddings as fallback
# ---------------------------------------------------------------------------
class DenseBranch:
    """Encapsulates 'embed these texts' behind one interface."""

    def __init__(self) -> None:
        self.backend = "unknown"
        try:
            from fastembed import TextEmbedding
            console.print("[dim]Initializing fastembed TextEmbedding(sentence-transformers/all-MiniLM-L6-v2)…[/dim]")
            self._model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            self.backend = "fastembed (ONNX)"
            console.print("[green]✓ fastembed ready — using ONNX dense embeddings.[/green]")
        except Exception as exc:  # noqa: BLE001 — offline / download / runtime errors
            console.print(Panel(
                f"[yellow]fastembed unavailable ({type(exc).__name__}: {exc})\n"
                "[dim]Falling back to shared local sentence-transformers "
                "(get_embeddings) for the dense branch.[/dim]",
                title="Dense fallback", border_style="yellow"))
            self._model = None
            self._lc_embeddings = get_embeddings()
            self.backend = "sentence-transformers fallback"

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        else:
            vecs = np.array(self._lc_embeddings.embed_documents(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)  # unit vectors -> dot = cosine


# ---------------------------------------------------------------------------
# 3. SPARSE BRANCH + RRF
# ---------------------------------------------------------------------------
def build_bm25(chunks: list[Document]) -> BM25Okapi:
    return BM25Okapi([tokenize(c.page_content) for c in chunks])


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(id) = Σ_lists 1/(k + rank), rank 1-based. k=60 dampens single-list bias."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------
def snippet(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def main() -> None:
    console.rule("[bold]MODULE 5 · APPROACH B · FASTEMBED HYBRID RETRIEVAL")

    chunks = build_chunks(load_raw_documents())
    console.print(f"[green]Corpus: {len(chunks)} chunks from data/samples/.[/green]")

    dense = DenseBranch()
    bm25 = build_bm25(chunks)

    chunk_vecs = dense.embed([c.page_content for c in chunks])
    chunk_ids = [c.metadata["id"] for c in chunks]
    by_id = {c.metadata["id"]: c for c in chunks}

    queries = [
        "How does RAG reduce hallucination?",
        "What are HNSW parameters like M and efSearch?",
        "Which vector database should I pick?",
    ]

    for q in queries:
        console.print(Panel(f"[bold white]QUERY:[/bold white] {q}", border_style="blue"))

        # --- dense top-10 (cosine) ---
        q_vec = dense.embed([q])[0]
        cos = chunk_vecs @ q_vec
        dense_order = np.argsort(-cos)[:10]
        dense_ids = [chunk_ids[i] for i in dense_order]
        dense_score = {chunk_ids[i]: float(cos[i]) for i in dense_order}

        # --- sparse top-10 (BM25) ---
        b_scores = bm25.get_scores(tokenize(q))
        sparse_order = np.argsort(-b_scores)[:10]
        sparse_ids = [chunk_ids[i] for i in sparse_order]
        sparse_score = {chunk_ids[i]: float(b_scores[i]) for i in sparse_order}

        # --- RRF fusion ---
        fused = rrf_fuse([dense_ids, sparse_ids], k=60)
        top5 = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:5]

        table = Table(title=f"Fused top-5  (dense={dense.backend} ⊕ BM25, RRF k=60)",
                      title_style="bold cyan", header_style="bold magenta")
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("rrf", justify="right", width=9)
        table.add_column("dense cos", justify="right", width=10)
        table.add_column("bm25", justify="right", width=8)
        table.add_column("source", width=24)
        table.add_column("snippet", overflow="fold")
        for rank, (doc_id, rrf_s) in enumerate(top5, 1):
            d = dense_score.get(doc_id)
            s = sparse_score.get(doc_id)
            table.add_row(
                str(rank), f"{rrf_s:.5f}",
                f"{d:.4f}" if d is not None else "—",
                f"{s:.3f}" if s is not None else "—",
                by_id[doc_id].metadata.get("source", "?"),
                snippet(by_id[doc_id].page_content),
            )
        console.print(table)
        console.print()

    console.rule("[bold]DONE")
    console.print(
        "[dim]Each row shows BOTH component scores: '—' means that doc was outside the "
        "branch's top-10, yet RRF can still surface it if the other branch ranks it well. "
        "That cross-branch rescue is the whole point of hybrid search.[/dim]"
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
        sys.exit(0)
