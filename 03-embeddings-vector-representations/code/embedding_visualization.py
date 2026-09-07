"""
Module 3 - Embedding visualization: do similar texts actually cluster?

Run from the project root:
    uv run python 03-embeddings-vector-representations/code/embedding_visualization.py

What this script does (NO LLM, NO API keys - local embeddings + NumPy + matplotlib):
  1. Embeds ~30 short RAG-domain texts grouped into 4 semantic categories
     (retrieval, vector stores, chunking, evaluation) plus cross-topic distractors.
  2. Reduces the high-dim vectors to 2D with PCA implemented in pure NumPy
     (center -> SVD -> top-2 components). No sklearn needed.
  3. Draws a scatter plot colored by category and saves it to
     03-embeddings-vector-representations/embedding_pca.png
  4. Prints the saved path + a quantitative observation about cluster separation
     (mean within-category vs between-category cosine similarity).

Why this matters: embeddings are only useful if "similar meaning = nearby
vector". This plot makes that assumption visible instead of assumed.
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
from rich.panel import Panel
from rich.table import Table

from shared.config import get_embeddings

console = Console()

# --------------------------------------------------------------------------- #
# ~30 short texts: 4 semantic categories + distractors
# --------------------------------------------------------------------------- #
TEXTS = [
    # Category A: Retrieval & search
    ("Retrieval finds relevant passages before generation.", "retrieval"),
    ("Semantic search matches meaning, not just keywords.", "retrieval"),
    ("A retriever returns the top-k most similar chunks.", "retrieval"),
    ("Hybrid retrieval blends dense vectors with BM25 scores.", "retrieval"),
    ("Recall at k measures how often the right doc is retrieved.", "retrieval"),
    ("Query expansion adds paraphrases to improve search.", "retrieval"),
    ("Reranking reorders candidates with a cross-encoder.", "retrieval"),
    ("Sparse vectors encode term frequencies for keyword match.", "retrieval"),
    # Category B: Vector stores & indexes
    ("A vector database stores embeddings for fast lookup.", "vector store"),
    ("HNSW builds a graph index for approximate nearest neighbors.", "vector store"),
    ("Cosine distance compares vector directions in the index.", "vector store"),
    ("Chroma persists vectors on disk for later queries.", "vector store"),
    ("An inverted index maps terms to the documents containing them.", "vector store"),
    ("Indexing trades memory for faster nearest-neighbor search.", "vector store"),
    ("pgvector stores embeddings inside a SQL database.", "vector store"),
    ("Sharding splits a huge vector index across machines.", "vector store"),
    # Category C: Chunking & preprocessing
    ("Chunking splits documents into smaller pieces.", "chunking"),
    ("Recursive splitting respects paragraph and sentence boundaries.", "chunking"),
    ("Overlap between chunks keeps context across boundaries.", "chunking"),
    ("Token limits cap how long each chunk may be.", "chunking"),
    ("Markdown headers guide where to cut a document.", "chunking"),
    ("Smaller chunks are precise but lose surrounding context.", "chunking"),
    ("Parent documents give small chunks more context at query time.", "chunking"),
    ("Preprocessing strips boilerplate before embedding text.", "chunking"),
    # Category D: Evaluation & quality
    ("RAGAS scores faithfulness of generated answers.", "evaluation"),
    ("Context recall checks whether the right facts were retrieved.", "evaluation"),
    ("An eval harness tracks quality across pipeline changes.", "evaluation"),
    ("LLM-as-judge grades answers against a rubric.", "evaluation"),
    ("MRR ranks results by where the first hit appears.", "evaluation"),
    ("A/B tests compare two retrieval pipelines on the same set.", "evaluation"),
    ("Ground truth labels are required for supervised metrics.", "evaluation"),
    ("Regression alerts fire when retrieval quality drops.", "evaluation"),
    # Distractors (deliberately off-topic)
    ("The souffle collapsed because the oven door opened early.", "distractor"),
    ("The striker scored a bicycle kick in stoppage time.", "distractor"),
    ("Quantum computers exploit superposition of qubits.", "distractor"),
]
CATEGORIES = ["retrieval", "vector store", "chunking", "evaluation", "distractor"]
COLORS = {
    "retrieval": "#1f77b4",
    "vector store": "#d62728",
    "chunking": "#2ca02c",
    "evaluation": "#9467bd",
    "distractor": "#7f7f7f",
}


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    """Reduce N x D vectors to N x 2 using PCA in pure NumPy.

    Steps: center the data, SVD-decompose it, keep the two largest
    singular directions. Equivalent to sklearn's PCA(n_components=2).
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # economy SVD: U (N x k), S (k,), Vt (k x D); projection = U * S
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * s[:2]


def separation_stats(vectors: np.ndarray, cats: list[str]) -> None:
    """Mean within-category vs between-category cosine similarity."""
    v = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    within, between = [], []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = float(v[i] @ v[j])
            (within if cats[i] == cats[j] else between).append(sim)
    w, b = float(np.mean(within)), float(np.mean(between))
    t = Table(title="Cluster separation (cosine similarity)", title_style="bold magenta")
    t.add_column("Pair type", style="bold cyan")
    t.add_column("mean cosine", justify="right")
    t.add_column("pairs", justify="right")
    t.add_row("within same category", f"{w:+.3f}", str(len(within)))
    t.add_row("between different categories", f"{b:+.3f}", str(len(between)))
    console.print(t)
    console.print(Panel(
        f"Within-category similarity ({w:+.3f}) exceeds between-category "
        f"({b:+.3f}) by [bold]{w - b:+.3f}[/bold].\n\n"
        "That gap is the whole promise of embeddings: semantically related texts land\n"
        "closer together than unrelated ones, which is exactly what a vector index\n"
        "exploits when it answers 'find the k nearest neighbours'. If the gap were ~0,\n"
        "the model would be useless for retrieval.",
        title="Observation", border_style="green"))


def main() -> None:
    console.rule("[bold cyan]Module 3 - Embedding Visualization (PCA scatter)[/]")

    texts = [t for t, _ in TEXTS]
    cats = [c for _, c in TEXTS]

    console.print(f"\n[bold]Step 1:[/bold] Embedding {len(texts)} texts with the shared local model...")
    try:
        emb = get_embeddings()
        vectors = np.array(emb.embed_documents(texts), dtype=np.float32)
    except Exception as exc:
        console.print(Panel(
            f"[red]Could not load the local embedding model:[/]\n{exc}\n\n"
            "[yellow]First run downloads all-MiniLM-L6-v2 (~90 MB) from Hugging Face.\n"
            "Check your internet connection, then re-run.[/]",
            title="Setup issue", border_style="red"))
        return
    console.print(f"  * {len(texts)} texts -> shape {vectors.shape}  ({vectors.shape[1]} dims)")

    console.print("\n[bold]Step 2:[/bold] PCA to 2D (pure NumPy: center -> SVD -> top-2 components)")
    pts = pca_2d(vectors)
    console.print(f"  * projected to shape {pts.shape}")

    console.print("\n[bold]Step 3:[/bold] Scatter plot colored by category")
    import matplotlib
    matplotlib.use("Agg")  # headless: render to file, no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    for cat in CATEGORIES:
        idx = [i for i, c in enumerate(cats) if c == cat]
        ax.scatter(pts[idx, 0], pts[idx, 1], c=COLORS[cat], label=cat,
                   s=70, alpha=0.85, edgecolors="white", linewidths=0.6)
    ax.set_title("Embedding space (PCA, 2D) - RAG-domain texts by category")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Category")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = pathlib.Path(__file__).resolve().parents[1] / "embedding_pca.png"
    fig.savefig(out_path)
    plt.close(fig)
    console.print(f"  [green]Saved plot ->[/] [bold]{out_path}[/bold]")

    console.print("\n[bold]Step 4:[/bold] How well do the categories separate?")
    separation_stats(vectors, cats)

    console.rule("[bold green]Done - open embedding_pca.png to see the clusters.[/]")


if __name__ == "__main__":
    main()
