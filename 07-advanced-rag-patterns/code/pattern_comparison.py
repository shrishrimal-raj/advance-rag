"""
Pattern Shootout: Naive vs Hybrid vs RAG-Fusion vs HyDE
=======================================================
Side-by-side comparison of four retrieval patterns on 3 fixed questions over the
small sample corpus (data/samples/):

    naive   — one dense search with the raw question
    hybrid  — BM25Okapi + dense, fused with RRF (k=60)
    fusion  — LLM expands query into 3 variants, dense each, fused with RRF (k=60)
    hyde    — LLM writes a hypothetical doc, embed it, dense search

Metrics per (question, pattern):
    latency ms            — retrieval stage only (search + fusion; excludes LLM calls)
    keyword coverage      — fraction of the question's content keywords that appear
                            in the retrieved top-k context (cheap relevance proxy)
    answer length         — words in the LLM answer grounded on that context

Run from project root:
    uv run python 07-advanced-rag-patterns/code/pattern_comparison.py
"""

import sys
import pathlib

# Windows console is cp1252; force UTF-8 output to avoid encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Project root (this file lives in <root>/07-advanced-rag-patterns/code/)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import re
import time

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

RRF_K = 60
TOP_K = 4                 # docs per list / final context size
NUM_VARIANTS = 3          # for RAG Fusion
QUESTIONS = [
    "How does vector-based retrieval work in a RAG system?",
    "What is chunking and why does chunk size matter?",
    "How do I make my search faster and more accurate?",
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "to", "and",
    "or", "how", "what", "why", "does", "do", "did", "my", "i", "it", "that",
    "this", "with", "for", "be", "as", "at", "by", "from", "you", "your",
}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def load_corpus() -> list[str]:
    data_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "samples"
    chunks: list[str] = []
    for file_path in sorted(data_dir.iterdir()):
        if file_path.suffix.lower() not in (".txt", ".md"):
            continue
        for para in file_path.read_text(encoding="utf-8").split("\n\n"):
            para = para.strip()
            if len(para) < 20:
                continue
            for j in range(0, len(para), 500):
                piece = para[j:j + 500].strip()
                if piece:
                    chunks.append(piece)
    return chunks


def build_store(chunks: list[str]):
    import chromadb
    from shared.config import get_embeddings

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(chunks)
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("pattern_comparison")
    collection.add(
        ids=[f"doc_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
    )
    return collection, embeddings


def dense_search(collection, embeddings, text: str, k: int = TOP_K) -> list[str]:
    vec = embeddings.embed_query(text)
    res = collection.query(query_embeddings=[vec], n_results=k)
    return res["ids"][0]


# ---------------------------------------------------------------------------
# RRF (numpy) — shared by hybrid and fusion
# ---------------------------------------------------------------------------
def rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    all_ids: list[str] = []
    seen: set[str] = set()
    for rl in rank_lists:
        for doc_id in rl:
            if doc_id not in seen:
                seen.add(doc_id)
                all_ids.append(doc_id)
    idx = {d: i for i, d in enumerate(all_ids)}
    scores = np.zeros(len(all_ids))
    for rl in rank_lists:
        for rank, doc_id in enumerate(rl, start=1):
            scores[idx[doc_id]] += 1.0 / (k + rank)
    order = np.argsort(-scores, kind="stable")
    return [all_ids[i] for i in order[:TOP_K]]


# ---------------------------------------------------------------------------
# The four patterns — each returns (doc_ids, retrieval_latency_ms)
# ---------------------------------------------------------------------------
def pattern_naive(collection, embeddings, question: str):
    t0 = time.perf_counter()
    ids = dense_search(collection, embeddings, question)
    return ids, (time.perf_counter() - t0) * 1000


def pattern_hybrid(collection, embeddings, bm25, question: str):
    t0 = time.perf_counter()
    dense_ids = dense_search(collection, embeddings, question)
    scores = bm25.get_scores(question)
    sparse_ids = [str(i) for i in np.argsort(-scores)[:TOP_K]]
    ids = rrf_fuse([dense_ids, sparse_ids])
    return ids, (time.perf_counter() - t0) * 1000


def pattern_fusion(collection, embeddings, llm_ok: bool, question: str):
    if llm_ok:
        try:
            from shared.config import get_llm
            prompt = f"""Given the question: "{question}"

Generate {NUM_VARIANTS} different ways to rephrase this question for document
retrieval. Output ONLY the {NUM_VARIANTS} queries, one per line, no numbering."""
            lines = [ln.strip() for ln in get_llm(temperature=0.7).invoke(prompt)
                     .content.strip().split("\n") if ln.strip()]
            variants = lines[:NUM_VARIANTS] or [question]
        except Exception:
            variants = None
    else:
        variants = None
    if not variants:  # graceful fallback: heuristic rephrasings
        words = [w for w in question.lower().split() if len(w) > 3]
        variants = [question, " ".join(words), f"explain {question}"]
    while len(variants) < NUM_VARIANTS:
        variants.append(question)
    # Latency = retrieval stage only (search + RRF); LLM expansion excluded.
    t0 = time.perf_counter()
    ids = rrf_fuse([dense_search(collection, embeddings, v) for v in variants])
    return ids, (time.perf_counter() - t0) * 1000


def pattern_hyde(collection, embeddings, llm_ok: bool, question: str):
    hyp_doc = question  # fallback = naive
    if llm_ok:
        try:
            from shared.config import get_llm
            prompt = f"""Write a HYPOTHETICAL ANSWER to this question as a technical
knowledge-base paragraph (3-5 sentences, domain terminology, no preamble).
It will be embedded and used as a search query.

Question: {question}

Hypothetical document:"""
            hyp_doc = get_llm(temperature=0.3).invoke(prompt).content.strip()
        except Exception:
            pass
    # Latency = retrieval stage only; hypothetical-doc generation excluded.
    t0 = time.perf_counter()
    ids = dense_search(collection, embeddings, hyp_doc)
    return ids, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def keyword_coverage(question: str, context: str) -> float:
    """Fraction of the question's content keywords present in the context."""
    q_words = {w for w in re.findall(r"[a-z0-9]+", question.lower())
               if w not in STOPWORDS and len(w) > 2}
    if not q_words:
        return 0.0
    ctx = context.lower()
    hits = sum(1 for w in q_words if w in ctx)
    return hits / len(q_words)


def ask_answer(llm, question: str, context: str) -> int:
    """Return word count of a short grounded answer; 0 if LLM unavailable."""
    prompt = f"""Using ONLY this context, answer in at most 3 sentences.
If the context lacks the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    try:
        response = llm.invoke(prompt)
        return len(response.content.split())
    except Exception as e:
        console.print(f"[yellow]Answer call failed ({type(e).__name__}); counting 0 words.[/yellow]")
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    console.print(Panel("[bold magenta]PATTERN SHOOTOUT[/bold magenta]",
                        subtitle="naive vs hybrid (BM25+dense RRF) vs RAG-Fusion vs HyDE"))

    # --- setup -------------------------------------------------------------
    chunks = load_corpus()
    console.print(f"Corpus: {len(chunks)} chunks from data/samples/")
    try:
        collection, embeddings = build_store(chunks)
    except Exception as e:
        console.print(f"[red]Embeddings/vector store failed: {e}[/red]")
        console.print("[yellow]Hint: local MiniLM should already be cached.[/yellow]")
        return 0

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([c.split() for c in chunks])

    # Probe LLM availability once up front (real ping, not just construction).
    llm_ok = True
    llm = None
    try:
        from shared.config import get_llm, get_llm_provider_name
        llm = get_llm(temperature=0.0)
        llm.invoke("Reply with the single word: ok")
        console.print(f"LLM: {get_llm_provider_name()}")
    except Exception as e:
        llm_ok = False
        console.print(f"[yellow]LLM unavailable ({type(e).__name__}) — running retrieval-only "
                      "mode; answers will be 'n/a'. Check YOLO_AUTO_API_KEY in .env.[/yellow]")

    # --- run the shootout ---------------------------------------------------
    rows = []
    for qi, question in enumerate(QUESTIONS, 1):
        console.print(f"\n[bold cyan]Question {qi}/{len(QUESTIONS)}:[/bold cyan] {question}")
        runners = {
            "naive":  lambda: pattern_naive(collection, embeddings, question),
            "hybrid": lambda: pattern_hybrid(collection, embeddings, bm25, question),
            "fusion": lambda: pattern_fusion(collection, embeddings, llm_ok, question),
            "hyde":   lambda: pattern_hyde(collection, embeddings, llm_ok, question),
        }
        for name, run in runners.items():
            doc_ids, latency_ms = run()
            context = "\n".join(collection.get(ids=doc_ids)["documents"])
            coverage = keyword_coverage(question, context)
            words = ask_answer(llm, question, context) if llm_ok else 0
            rows.append((qi, name, latency_ms, coverage, words))
            console.print(f"  [green]{name:<7}[/green] {latency_ms:8.1f} ms | "
                          f"coverage {coverage:.0%} | answer {words if llm_ok else 'n/a'} words")

    # --- results table ------------------------------------------------------
    table = Table(title="Pattern Shootout Results", show_lines=True)
    table.add_column("Q#", width=4)
    table.add_column("Pattern", width=8)
    table.add_column("Retrieval (ms)", justify="right")
    table.add_column("Keyword coverage", justify="right")
    table.add_column("Answer (words)", justify="right")
    for qi, name, latency_ms, coverage, words in rows:
        style = {"naive": "dim", "hybrid": "white", "fusion": "cyan", "hyde": "magenta"}[name]
        table.add_row(str(qi), f"[{style}]{name}[/{style}]", f"{latency_ms:.1f}",
                      f"{coverage:.0%}", str(words) if llm_ok else "n/a")
    console.print(table)

    # --- verdict -------------------------------------------------------------
    verdict = """
[bold]Verdict — when each pattern wins:[/bold]

• [dim]naive[/dim]     — cheapest and fastest. Wins when queries already use the corpus's
  vocabulary and the corpus is small/clean. Baseline to beat.
• [bold]hybrid[/bold]  — best accuracy-per-ms: BM25 catches exact terms (names, acronyms,
  error codes) that dense vectors blur, and RRF merges both lists. The default
  production choice when you can't afford extra LLM calls.
• [bold]fusion[/bold]  — wins on ambiguous or underspecified questions where one phrasing
  misses. Costs one extra LLM call (query expansion) but retrieval itself stays cheap.
• [bold]hyde[/bold]    — wins on vocabulary mismatch (casual user, technical corpus).
  Costs one LLM call before retrieval; can hurt on very specific factual queries
  where the hypothetical doc drifts off-topic.

Rule of thumb: start naive, move to hybrid, add fusion/HyDE only when you've
measured a specific failure mode they fix. Every extra LLM call is latency + cost.
"""
    console.print(Panel(verdict, title="[green]Verdict[/green]", border_style="green"))
    console.print("[bold green]Done.[/bold green]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
