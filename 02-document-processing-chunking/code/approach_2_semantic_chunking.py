"""Module 2 — Approach B: Semantic Chunking FROM SCRATCH (no LangChain splitters).

Idea: split text into sentences, embed each sentence, then walk the sequence
computing cosine similarity between CONSECUTIVE sentences. Where similarity
dips below a threshold, the topic has changed -> cut a chunk boundary there.

This is the classic "topic shift detection" approach used by semantic-chunkers
in production RAG stacks. Here we build it with pure numpy + get_embeddings().

Run from project root:
    uv run python 02-document-processing-chunking/code/approach_2_semantic_chunking.py
"""
import sys
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import re

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
SAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "samples"


# ------------------------------------------------------------------ helpers
def split_sentences(text: str) -> list[str]:
    """Naive but robust sentence splitter: break on .!? followed by space+capital/EOL."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def cosine_similarities(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between each consecutive pair of rows."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    sims = np.einsum("ij,ij->i", unit[:-1], unit[1:])
    return sims


def pick_threshold(sims: np.ndarray) -> float:
    """Adaptive threshold: mean - 0.5 * std, clamped to [0.2, 0.95]."""
    if len(sims) == 0:
        return 0.75
    thr = float(np.mean(sims) - 0.5 * np.std(sims))
    return float(min(max(thr, 0.2), 0.95))


def semantic_chunk(sentences: list[str]):
    """Return (groups of sentence indices, similarity trace, threshold)."""
    from shared.config import get_embeddings

    emb = get_embeddings()
    vectors = np.array(emb.embed_documents(sentences), dtype=np.float32)
    sims = cosine_similarities(vectors)
    threshold = pick_threshold(sims)

    groups: list[list[int]] = [[0]]
    for i, s in enumerate(sims):
        if s < threshold:
            groups.append([i + 1])
        else:
            groups[-1].append(i + 1)
    return groups, sims, threshold


# ------------------------------------------------------------------ main flow
def main():
    console.print(Panel("[bold]Module 2 — Approach B: Semantic Chunking (from scratch)[/bold]",
                        border_style="blue"))

    text_path = SAMPLES_DIR / "rag_overview.txt"
    text = text_path.read_text(encoding="utf-8")
    console.print(f"[dim]Source: {text_path.name} ({len(text)} chars)[/dim]\n")

    sentences = split_sentences(text)
    console.print(f"[bold]1) Sentence split:[/bold] {len(sentences)} sentences\n")

    # Graceful degradation: embeddings unavailable -> clear hint, exit 0
    try:
        groups, sims, threshold = semantic_chunk(sentences)
    except Exception as e:
        console.print(Panel(
            "[yellow]Embedding model unavailable.[/yellow]\n\n"
            f"Error: {e}\n\n"
            "Semantic chunking needs local embeddings (sentence-transformers MiniLM).\n"
            "Fix: make sure the model cache exists (run any module using get_embeddings()\n"
            "once while online) or set EMBEDDING_PROVIDER=openai with OPENAI_API_KEY.",
            title="⚠ Cannot continue", border_style="yellow"))
        sys.exit(0)

    # ------------------------------------------------- 2) show similarity trace
    console.rule("[bold]2) Consecutive-sentence cosine similarity[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_column()
    for i, s in enumerate(sims):
        marker = "  [red]✂ CUT[/red]" if s < threshold else ""
        bar = "█" * int(round(s * 20))
        table.add_row(f"s{i}→s{i+1}", f"{s:.3f}", f"[green]{bar}[/green]{marker}")
    console.print(table)
    console.print(f"\nThreshold (mean − 0.5·std, adaptive): [bold]{threshold:.3f}[/bold]")

    # ------------------------------------------------- 3) resulting chunks
    console.rule("[bold]3) Semantic chunks (boundaries marked)[/bold]")
    for ci, grp in enumerate(groups):
        body = " ".join(sentences[i] for i in grp)
        console.print(Panel(
            body,
            title=f"[cyan]Chunk {ci}[/cyan]  [dim]sentences {grp[0]}–{grp[-1]} · {len(body)} chars[/dim]",
            border_style="cyan", expand=False))

    # ------------------------------------------------- 4) compare vs RecursiveCharacterTextSplitter
    console.rule("[bold]4) Comparison vs RecursiveCharacterTextSplitter[/bold]")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    rc_chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_text(text)

    sem_chars = [len(" ".join(sentences[i] for i in g)) for g in groups]
    rc_chars = [len(c) for c in rc_chunks]

    t = Table(title="Semantic (this script) vs RecursiveCharacterTextSplitter")
    t.add_column("Metric", style="cyan")
    t.add_column("Semantic chunking", justify="right")
    t.add_column("Recursive (300/50)", justify="right")
    t.add_row("Chunk count", str(len(groups)), str(len(rc_chunks)))
    t.add_row("Avg chunk chars", str(round(sum(sem_chars) / len(sem_chars))),
              str(round(sum(rc_chars) / len(rc_chars))))
    t.add_row("Min / Max chars", f"{min(sem_chars)} / {max(sem_chars)}",
              f"{min(rc_chars)} / {max(rc_chars)}")
    t.add_row("Cuts respect sentence ends", "always", "only when size allows")
    t.add_row("Cuts respect topic shifts", "yes (similarity dip)", "no (size-driven)")
    console.print(t)

    # Boundary positions in the original text
    def offsets(chunks_texts):
        pos, out = 0, []
        for c in chunks_texts:
            idx = text.find(c, pos)
            out.append(idx if idx >= 0 else pos)
            pos = idx + len(c) if idx >= 0 else pos
        return out

    sem_texts = [" ".join(sentences[i] for i in g) for g in groups]
    console.print("\n[dim]Boundary char offsets in source text:[/dim]")
    console.print(f"  semantic : {offsets(sem_texts)}")
    console.print(f"  recursive: {offsets(rc_chunks)}")

    console.print(
        "\n[bold red]Takeaway:[/bold red] recursive splitting optimizes for SIZE; "
        "semantic splitting optimizes for TOPIC COHERENCE. Semantic chunks cost one "
        "embedding pass per sentence (fine offline at ingestion time) and produce "
        "chunks that answer questions standalone — the granularity test from 02-learning.md."
    )


if __name__ == "__main__":
    main()
