"""
Module 5 — Approach C: Reciprocal Rank Fusion FROM SCRATCH (pure numpy)
=======================================================================
No LangChain retrievers, no vector store. Just:

    dense scores : cosine similarity between query and chunk embeddings
                   (local MiniLM via shared.config.get_embeddings + numpy dot)
    sparse scores: rank_bm25.BM25Okapi
    fusion       : RRF implemented in ~10 lines of numpy, k = 60

For each sample query we print the dense-only ranking, the BM25-only
ranking, and the fused ranking side by side, then explain WHY fusion wins
the tie/diversity cases you can see in the output.

No LLM required. Run from the project root:
    uv run python 05-basic-retrieval-techniques/code/approach_3_rrf_from_scratch.py
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
# 1. RRF FROM SCRATCH — the whole algorithm in numpy
# ---------------------------------------------------------------------------
def rrf_from_scratch(ranked_lists: list[list[int]], n_docs: int, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion over ranked lists of DOCUMENT INDICES.

    score[i] = Σ_lists 1 / (k + rank_of_i_in_list)      (rank is 1-based;
    docs absent from a list contribute 0 for that list).

    Why ranks instead of raw scores? Dense cosine lives in [0,1], BM25 is
    unbounded — the scales are incomparable. Rank is the only quantity both
    lists share, so fusing ranks needs zero calibration. k=60 (the paper's
    default) makes rank 1 vs rank 2 nearly equal (1/61 vs 1/62), so no single
    branch can dominate the fusion.
    """
    scores = np.zeros(n_docs, dtype=np.float64)
    for ranked in ranked_lists:
        # 1/(k + rank) for each position, scattered onto the doc indices.
        # np.add.at accumulates when the same doc appears in several lists.
        inv = 1.0 / (k + np.arange(1, len(ranked) + 1))
        np.add.at(scores, np.asarray(ranked, dtype=np.int64), inv)
    return scores


# ---------------------------------------------------------------------------
# 2. CORPUS + SCORES
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


def embed_unit(chunks: list[Document], emb) -> np.ndarray:
    """Unit-normalized chunk vectors so dot product == cosine similarity."""
    vecs = np.array(emb.embed_documents(
        [c.page_content for c in chunks]), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def snippet(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


# ---------------------------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    console.rule("[bold]MODULE 5 · APPROACH C · RRF FROM SCRATCH (NUMPY)")

    chunks = load_chunks()
    n = len(chunks)
    console.print(f"[green]Corpus: {n} chunks from data/samples/.[/green]")

    emb = get_embeddings()
    chunk_vecs = embed_unit(chunks, emb)
    bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    queries = [
        "How does RAG reduce hallucination?",
        "What are HNSW parameters like M and efSearch?",
        "Which vector database should I pick?",
    ]

    for q in queries:
        console.print(Panel(f"[bold white]QUERY:[/bold white] {q}", border_style="blue"))

        # --- branch 1: dense cosine ---
        q_vec = embed_unit([Document(page_content=q)], emb)[0]
        cos = chunk_vecs @ q_vec
        dense_top = np.argsort(-cos)[:5]

        # --- branch 2: BM25 ---
        b_scores = bm25.get_scores(tokenize(q))
        sparse_top = np.argsort(-b_scores)[:5]

        # --- fusion: pure-numpy RRF over the two ranked index lists ---
        fused = rrf_from_scratch([list(dense_top), list(sparse_top)], n_docs=n, k=60)
        fused_top = np.argsort(-fused)[:5]

        table = Table(title=f"Dense-only   |   BM25-only   |   Fused (RRF k=60)",
                      title_style="bold cyan", header_style="bold magenta")
        table.add_column("dense #", justify="right", width=7)
        table.add_column("dense chunk", overflow="fold")
        table.add_column("bm25 #", justify="right", width=7)
        table.add_column("bm25 chunk", overflow="fold")
        table.add_column("rrf #", justify="right", width=6)
        table.add_column("fused chunk", overflow="fold")
        table.add_column("rrf score", justify="right", width=9)
        for r in range(5):
            d, s, f = int(dense_top[r]), int(sparse_top[r]), int(fused_top[r])
            table.add_row(
                str(r + 1), f"{cos[d]:.3f} · {snippet(chunks[d].page_content)}",
                str(r + 1), f"{b_scores[s]:.2f} · {snippet(chunks[s].page_content)}",
                str(r + 1), f"{snippet(chunks[f].page_content)}",
                f"{fused[f]:.5f}",
            )
        console.print(table)

        # --- why did fusion win? annotate the fused top-5 with per-branch ranks ---
        dense_rank = {int(i): p + 1 for p, i in enumerate(dense_top)}
        sparse_rank = {int(i): p + 1 for p, i in enumerate(sparse_top)}
        notes = []
        for f in fused_top:
            f = int(f)
            dr, sr = dense_rank.get(f, "—"), sparse_rank.get(f, "—")
            notes.append(f"#{f} dense={dr} bm25={sr}")
        console.print("[dim]" + "  |  ".join(notes) + "[/dim]")
        console.print()

    console.rule("[bold]DONE")
    console.print(Panel(
        "[bold]Why fusion wins ties & diversity cases[/bold]\n"
        "• A doc ranked #1 by dense but #4 by BM25 scores 1/61 + 1/64 ≈ 0.0323,\n"
        "  beating a doc that is #1 in only ONE list (1/61 ≈ 0.0164). Consensus\n"
        "  across branches beats single-branch conviction.\n"
        "• Ties: two docs with near-identical cosine (e.g. 0.621 vs 0.620) are\n"
        "  indistinguishable to the dense branch alone; BM25's term evidence\n"
        "  breaks the tie deterministically.\n"
        "• Diversity: dense tends to return paraphrases of the same idea;\n"
        "  BM25 pulls in chunks matched by exact terms (names, codes, params)\n"
        "  that embeddings blur — the fused list covers both angles.\n"
        "• Scale-free: cosine ∈ [0,1] and BM25 ∈ [0,∞) are never compared\n"
        "  directly — only their RANKS enter the formula, so no calibration.",
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
