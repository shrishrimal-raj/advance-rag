"""
Module 3 - Embeddings & Vector Representations: hands-on lab.

Run from the project root:
    uv run python 03-embeddings-vector-representations/code/main.py

What this lab does (NO LLM, NO API keys required - local embeddings + NumPy only):
  1. Embed ~8 short sentences across 3 topics (RAG, cooking, sports).
  2. Compute pairwise COSINE, L2 (Euclidean) and INNER-PRODUCT matrices with NumPy.
  3. Prove that cosine similarity == inner product once vectors are L2-normalized.
  4. Do a nearest-neighbour lookup for a fresh query sentence.
  5. Print the embedding dimensionality + a note on model choice.

NOTE: console output is kept pure ASCII so it renders on ANY terminal/locale
      (Windows cp1252, macOS, Linux) without UnicodeEncodeError.
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib

# Make the project root importable so we can reuse shared.config from any folder.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from shared.config import get_embeddings

console = Console()


# --------------------------------------------------------------------------- #
# Small, explicit metric helpers (so the math is visible, not hidden in a lib)
# --------------------------------------------------------------------------- #
def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit-length version of a vector (or a batch of vectors)."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)  # guard against div-by-zero
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(a,b) = (a.b) / (|a| |b|)  ->  range [-1, 1]. Direction only."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance |a - b|. Smaller = more similar."""
    return float(np.linalg.norm(a - b))


def inner_product(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product a.b. Larger = more aligned (magnitude-dependent until normalized)."""
    return float(np.dot(a, b))


def pairwise_matrix(vectors: np.ndarray, fn) -> np.ndarray:
    """Fill an N x N matrix applying `fn` to every pair of rows."""
    n = len(vectors)
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            m[i, j] = fn(vectors[i], vectors[j])
    return m


def show_matrix(title: str, m: np.ndarray, labels, fmt: str, note: str = "") -> None:
    """Render a square similarity/distance matrix as a readable rich table."""
    t = Table(title=title, title_style="bold magenta", header_style="bold")
    t.add_column("", style="dim", width=4)
    for lab in labels:
        t.add_column(lab, justify="right")
    for i, row_lab in enumerate(labels):
        t.add_row(row_lab, *[fmt.format(m[i, j]) for j in range(len(labels))])
    console.print(t)
    if note:
        console.print(f"  [dim]{note}[/]")


def main() -> None:
    console.rule("[bold cyan]Module 3 - Embeddings & Vector Representations[/]")

    # ---- Step 1: sentences across 3 topics (+ distractors) ------------------
    sentences = [
        # Topic A: RAG / AI
        "Retrieval augmented generation grounds answers in external documents.",
        "A vector database stores embeddings for fast semantic search.",
        # Topic B: Cooking
        "Simmer the tomato sauce until it thickens and turns deep red.",
        "Fold the egg whites gently to keep the souffle light and airy.",
        # Topic C: Sports
        "The striker scored a bicycle kick in the final minute of the match.",
        "The marathon runner crossed the finish line under two hours.",
        # Cross-topic distractors
        "The stock market rallied sharply after the earnings report.",
        "Quantum computers use qubits that can exist in superposition.",
    ]
    labels = [f"S{i + 1}" for i in range(len(sentences))]

    # ---- Step 2: embed ------------------------------------------------------
    console.print("\n[bold]Step 1:[/bold] Embedding sentences with the shared local model...")
    try:
        emb = get_embeddings()
        vectors = np.array(emb.embed_documents(sentences), dtype=np.float32)
    except Exception as exc:  # first run downloads the model; be helpful if it fails
        console.print(Panel(
            f"[red]Could not load the local embedding model:[/]\n{exc}\n\n"
            "[yellow]First run downloads all-MiniLM-L6-v2 (~90 MB) from Hugging Face.\n"
            "Check your internet connection, then re-run.[/]",
            title="Setup issue", border_style="red"))
        return
    dim = vectors.shape[1]
    console.print(f"  * {len(sentences)} sentences -> shape {vectors.shape}  ({dim} dims)")

    # ---- Step 3: pairwise matrices -----------------------------------------
    console.print("\n[bold]Step 2:[/bold] Pairwise similarity / distance matrices")
    cos_m = pairwise_matrix(vectors, cosine_similarity)
    l2_m = pairwise_matrix(vectors, l2_distance)
    ip_m = pairwise_matrix(vectors, inner_product)

    show_matrix("Cosine similarity  (-1 .. 1, higher = closer)", cos_m, labels, "{:+.3f}",
                "Diagonal = 1.0 (a vector matches itself). Same-topic pairs cluster high.")
    show_matrix("L2 / Euclidean distance  (0 .. inf, lower = closer)", l2_m, labels, "{:.3f}",
                "Diagonal = 0.0. Same-topic pairs have smaller distances.")
    show_matrix("Inner product (dot)  (unbounded)", ip_m, labels, "{:+.3f}",
                "Raw dot product - magnitude-dependent, not yet a clean similarity.")

    # ---- Step 4: cosine == inner product after normalization ----------------
    console.print("\n[bold]Step 3:[/bold] Cosine ~ inner product after L2-normalization")
    normed = l2_normalize(vectors)
    ip_normed = normed @ normed.T  # inner products computed on unit-length vectors
    max_diff = float(np.max(np.abs(ip_normed - cos_m)))
    console.print(Panel(
        f"After normalizing every vector to unit length, the inner-product matrix\n"
        f"becomes IDENTICAL to the cosine-similarity matrix.\n\n"
        f"max |inner_product_normalized - cosine| = {max_diff:.2e}\n"
        f"(~ 0 -> they are the same operation once magnitudes are removed.)",
        title="Why this matters", border_style="green"))
    console.print("  [dim]This is why most pipelines normalize before storing - then a plain "
                  "dot product IS cosine similarity.[/]")

    # ---- Step 5: nearest-neighbour lookup ----------------------------------
    console.print("\n[bold]Step 4:[/bold] Nearest-neighbour lookup for a query")
    # A query that paraphrases the RAG sentences -> a RAG sentence should rank near the top.
    query = "How does retrieval augmented generation work?"
    qv = np.array(emb.embed_query(query), dtype=np.float32)
    sims = np.array([cosine_similarity(qv, v) for v in vectors])
    order = list(np.argsort(-sims))  # indices sorted by descending similarity
    top_k = 3
    t = Table(title=f"Top-{top_k} neighbours for: '{query}'", title_style="bold cyan")
    t.add_column("#", justify="right", style="dim")
    t.add_column("cosine", justify="right")
    t.add_column("sentence")
    for rank, idx in enumerate(order[:top_k], start=1):
        t.add_row(str(rank), f"{sims[idx]:+.3f}", sentences[idx])
    console.print(t)
    # Informational: where did the two RAG/AI sentences (S1, S2) land in the ranking?
    rag_ranks = [order.index(i) + 1 for i in (0, 1)]
    console.print(f"  [dim]RAG/AI sentences S1 & S2 ranked #{rag_ranks[0]} and #{rag_ranks[1]} - "
                  f"both near the top, as expected for a RAG query.[/]")

    # ---- Step 6: dimension + model note ------------------------------------
    console.print("\n[bold]Step 5:[/bold] Model & dimensionality")
    console.print(Panel(
        f"Embedding dimension : [bold]{dim}[/]\n"
        f"Model               : local sentence-transformers (all-MiniLM-L6-v2)\n"
        f"Cost                : free (runs locally, no API key)\n\n"
        "[dim]Trade-off: 384-dim MiniLM is fast & cheap but weaker than 1024-dim BGE/E5/GTE "
        "or 1536-dim OpenAI text-embedding-3-large. Pick by your quality-vs-cost budget "
        "(see 02-learning.md section 6 comparison table).[/]",
        title="Model choice", border_style="yellow"))

    console.rule("[bold green]Lab complete [OK][/]")


if __name__ == "__main__":
    main()
