"""Module 2 — Chunking Benchmark: size x overlap sweep on the sample corpus.

NO LLM, NO EMBEDDINGS needed. Sweeps chunk sizes (256/512/1024/2048 chars) x
overlaps (0% / 25%) over every file in data/samples/ and measures:

  • chunk count & average chunk length
  • sentence-boundary-cut ratio  — % of chunks that start MID-sentence
  • retrieval proxy score         — keyword-overlap F1 between the top-1 chunk and
                                    a set of sample questions (higher = better)

Ends with a written conclusion on the sweet spot.

Run from project root:
    uv run python 02-document-processing-chunking/code/chunking_benchmark.py
"""
import re
import sys
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
SAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "samples"

CHUNK_SIZES = [256, 512, 1024, 2048]
OVERLAPS = [0, 0.25]

# Questions grounded in the actual sample corpus (retrieval proxy).
QUESTIONS = [
    "What is a vector database used for?",
    "How does HNSW indexing work?",
    "How does RAG reduce hallucination?",
    "What are the distance metrics for embeddings?",
    "Which products are in the catalog?",
    "What does the company do?",
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for", "to",
    "and", "or", "what", "how", "which", "does", "do", "it", "its", "with",
}


def tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def load_corpus() -> dict[str, str]:
    """Raw text per sample file (raw text keeps the benchmark loader-agnostic)."""
    corpus = {}
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix in {".txt", ".md", ".csv", ".json"}:
            corpus[path.name] = path.read_text(encoding="utf-8")
    return corpus


def sentence_starts(text: str) -> set[str]:
    """First ~40 chars of each sentence — used to detect mid-sentence chunk starts."""
    starts = set()
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if len(s) >= 8:
            starts.add(s[:40])
    return starts


def mid_sentence_ratio(text: str, chunks: list[str]) -> float:
    """Fraction of chunks whose start does NOT match any sentence start."""
    if not chunks:
        return 0.0
    starts = sentence_starts(text)
    bad = 0
    for c in chunks:
        head = c.strip()[:40]
        if not any(head == s or head.startswith(s) or s.startswith(head) for s in starts):
            bad += 1
    return bad / len(chunks)


def retrieval_proxy(text: str, chunks: list[str], questions: list[str]) -> float:
    """Mean keyword-overlap F1 between each question and its best-matching chunk.

    F1 balances recall (does the chunk cover the question's terms?) against
    precision (is the chunk focused on them?) — large topic-diluted chunks lose
    precision, mid-sentence fragments lose recall.
    """
    scores = []
    for q in questions:
        q_tokens = tokenize(q)
        if not q_tokens:
            continue
        best = 0.0
        for c in chunks:
            c_tokens = tokenize(c)
            if not c_tokens:
                continue
            inter = len(q_tokens & c_tokens)
            if inter == 0:
                continue
            p = inter / len(c_tokens)
            r = inter / len(q_tokens)
            best = max(best, 2 * p * r / (p + r))
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def main():
    console.print(Panel("[bold]Module 2 — Chunking Benchmark (size × overlap sweep)[/bold]",
                        border_style="blue"))

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    corpus = load_corpus()
    console.print(f"[dim]Corpus: {', '.join(corpus)} ({sum(len(t) for t in corpus.values())} chars total)[/dim]\n")

    rows = []
    for size in CHUNK_SIZES:
        for ov in OVERLAPS:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=int(size * ov))
            n_chunks, n_chars, mid_ratios, proxies = 0, 0, [], []
            for name, text in corpus.items():
                chunks = splitter.split_text(text)
                n_chunks += len(chunks)
                n_chars += sum(len(c) for c in chunks)
                mid_ratios.append(mid_sentence_ratio(text, chunks))
                proxies.append(retrieval_proxy(text, chunks, QUESTIONS))
            rows.append({
                "size": size,
                "overlap": f"{int(ov * 100)}%",
                "chunks": n_chunks,
                "avg": round(n_chars / n_chunks) if n_chunks else 0,
                "mid": 100 * sum(mid_ratios) / len(mid_ratios),
                "proxy": 100 * sum(proxies) / len(proxies),
            })

    table = Table(title=f"Benchmark — {len(QUESTIONS)} proxy questions, {len(corpus)} files")
    table.add_column("Chunk size", justify="right", style="cyan")
    table.add_column("Overlap", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Avg chars", justify="right")
    table.add_column("Mid-sentence cuts", justify="right")
    table.add_column("Retrieval proxy", justify="right")
    best_proxy = max(r["proxy"] for r in rows)
    for r in rows:
        style = "[green]" if r["proxy"] == best_proxy else ""
        table.add_row(
            str(r["size"]), r["overlap"], str(r["chunks"]), str(r["avg"]),
            f"{r['mid']:.0f}%", f"{style}{r['proxy']:.0f}%[/green]" if style else f"{r['proxy']:.0f}%")
    console.print(table)

    # ------------------------------------------------------------- conclusion
    console.rule("[bold red]Conclusion — where's the sweet spot?[/bold red]")
    by_size = {}
    for r in rows:
        by_size.setdefault(r["size"], []).append(r)
    lines = []
    for size, rs in by_size.items():
        zero, ov25 = rs[0], rs[1]
        delta_mid = zero["mid"] - ov25["mid"]
        lines.append(
            f"• size={size:<5} overlap 0%→25%: mid-sentence cuts {zero['mid']:.0f}%→{ov25['mid']:.0f}% "
            f"({'−' if delta_mid >= 0 else '+'}{abs(delta_mid):.0f} pts), "
            f"proxy {zero['proxy']:.0f}%→{ov25['proxy']:.0f}%, chunks {zero['chunks']}→{ov25['chunks']}")
    console.print("\n".join(lines))

    best = max(rows, key=lambda r: r["proxy"])
    worst = min(rows, key=lambda r: r["proxy"])
    console.print(
        f"\n[bold green]Best config on this corpus:[/bold green] chunk_size={best['size']}, "
        f"overlap={best['overlap']} — proxy {best['proxy']:.0f}% "
        f"(worst: {worst['size']}/{worst['overlap']} at {worst['proxy']:.0f}%).\n"
        "Reading the trade-offs:\n"
        "• [bold]Smaller chunks[/bold] cut more sentences in half (high mid-sentence %) but give the\n"
        "  retriever tighter, higher-precision passages when they do match.\n"
        "• [bold]Larger chunks[/bold] keep sentences intact but dilute relevance — one chunk covering\n"
        "  two topics loses precision in the F1 proxy.\n"
        "• [bold]Overlap[/bold] trades storage/duplicate noise against boundary safety; its effect here\n"
        "  is small because RecursiveCharacterTextSplitter already prefers sentence seams.\n"
        "• Caveat: this corpus is only ~4 KB, so absolute scores are low and close together.\n"
        "  On real corpora the same sweep separates configs much more clearly — re-run this\n"
        "  script after pointing SAMPLES_DIR at your production docs."
    )


if __name__ == "__main__":
    main()
