"""A/B pipeline comparison — naive dense retrieval vs. hybrid BM25+dense RRF + rerank.

Pipeline A (baseline): single dense retriever (in-memory Chroma, MiniLM).
Pipeline B (challenger): hybrid retrieval — BM25Okapi + dense search fused with
    Reciprocal Rank Fusion (RRF), then a lightweight embedding-similarity
    rerank stage before the top-k are kept.

Both pipelines share the SAME 5 questions and the SAME LLM generator. Outputs
are scored with the custom metrics from approach_2_custom_metrics.py
(faithfulness proxy, context precision/recall, answer relevancy) — no RAGAS.

Run from the project root:
    uv run python 09-rag-evaluation-ragas/code/ab_pipeline_comparison.py

Needs: local embeddings (cached MiniLM) + an LLM for generation
(YOLO_AUTO_API_KEY / OPENAI_API_KEY / Ollama). Degrades gracefully.
"""
import sys, pathlib, json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- Course convention: make the project root importable --------------------
MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]     # 09-rag-evaluation-ragas
PROJECT_ROOT = MODULE_DIR.parent                             # advance-rag
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))  # sibling scripts

from shared.config import get_llm, get_embeddings, get_llm_provider_name  # noqa: E402
from approach_2_custom_metrics import (  # noqa: E402
    EmbedCache, score_example, load_eval_set, METRIC_NAMES)
from rich.console import Console          # noqa: E402
from rich.table import Table              # noqa: E402
from rich.panel import Panel              # noqa: E402
from rich import box                      # noqa: E402

console = Console()

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
N_QUESTIONS = 5
TOP_K = 5          # contexts fed to the generator
HYBRID_K = 10      # candidates per list before RRF fusion
RRF_K = 60         # standard RRF constant


# --------------------------------------------------------------------------- #
# Corpus                                                                      #
# --------------------------------------------------------------------------- #
def build_corpus():
    """Load sample docs, chunk them, embed into in-memory Chroma."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    docs = [Document(page_content=f.read_text(encoding="utf-8"),
                     metadata={"source": f.name})
            for f in sorted(SAMPLES_DIR.iterdir()) if f.is_file()]
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs)
    emb = get_embeddings()
    try:
        return Chroma.from_documents(chunks, emb), [c.page_content for c in chunks]
    except TypeError:
        return Chroma.from_documents(chunks, embedding_function=emb), \
               [c.page_content for c in chunks]


# --------------------------------------------------------------------------- #
# Pipeline A — naive dense                                                    #
# --------------------------------------------------------------------------- #
def retrieve_dense(vectorstore, question: str) -> list[str]:
    return [d.page_content for d in vectorstore.similarity_search(question, k=TOP_K)]


# --------------------------------------------------------------------------- #
# Pipeline B — hybrid BM25 + dense, RRF fusion, embedding rerank              #
# --------------------------------------------------------------------------- #
class HybridRetriever:
    def __init__(self, vectorstore, corpus_texts: list[str], ec: EmbedCache):
        from rank_bm25 import BM25Okapi
        self._vs = vectorstore
        self._ec = ec
        self._bm25 = BM25Okapi([t.split() for t in corpus_texts])
        self._corpus = corpus_texts

    def _dense_ranks(self, question: str) -> list[int]:
        # NOTE: similarity_search (not ..._with_relevance_scores) — MiniLM
        # relevance scores are not normalized to [0,1] and trigger warnings;
        # we only need the ranking order.
        docs = self._vs.similarity_search(question, k=HYBRID_K)
        ranks = []
        for d in docs:
            try:
                ranks.append(self._corpus.index(d.page_content))
            except ValueError:
                continue
        return ranks

    def _bm25_ranks(self, question: str) -> list[int]:
        scores = self._bm25.get_scores(question.split())
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in order if scores[i] > 0][:HYBRID_K]

    @staticmethod
    def _rrf(rank_lists: list[list[int]]) -> dict[int, float]:
        """Reciprocal Rank Fusion: score(d) = sum over lists of 1/(RRF_K + rank)."""
        fused: dict[int, float] = {}
        for ranks in rank_lists:
            for rank, doc_id in enumerate(ranks, start=1):
                if doc_id < 0:
                    continue
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        return fused

    def retrieve(self, question: str) -> list[str]:
        dense_ids = self._dense_ranks(question)
        bm25_ids = self._bm25_ranks(question)
        fused = self._rrf([dense_ids, bm25_ids])
        candidates = sorted(fused, key=lambda d: -fused[d])[:HYBRID_K]
        # Rerank stage: re-score candidates by direct query-doc embedding similarity.
        qv = self._ec.vec(question)
        scored = [(d, self._ec.cosine(qv, self._ec.vec(self._corpus[d]))) for d in candidates]
        scored.sort(key=lambda x: -x[1])
        return [self._corpus[d] for d, _ in scored[:TOP_K]]


# --------------------------------------------------------------------------- #
# Shared generator                                                            #
# --------------------------------------------------------------------------- #
PROMPT = """Answer the question using ONLY the provided context.
If the answer is not in the context, say exactly: "I don't know."
Be concise and factual.

Context:
{context}

Question: {question}
Answer:"""


def make_generator():
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm(temperature=0.0)

    def generate(question: str, contexts: list[str]) -> str:
        resp = chain.invoke({"context": "\n\n".join(contexts), "question": question})
        return resp.content if hasattr(resp, "content") else str(resp)
    return generate


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    console.print(Panel.fit(
        "[bold]A/B pipeline comparison[/bold]\n"
        "[cyan]A:[/cyan] naive dense (Chroma + MiniLM)\n"
        "[cyan]B:[/cyan] hybrid BM25+dense → RRF → embedding rerank\n"
        f"Same {N_QUESTIONS} questions · same generator "
        f"([cyan]{get_llm_provider_name()}[/cyan]) · scored with custom metrics",
        title="🆚 A/B Pipeline Comparison", border_style="green"))

    try:
        embedder = get_embeddings()
        _ = embedder.embed_query("warmup")
    except Exception as e:
        console.print(Panel(
            f"[red]Local embeddings unavailable:[/red] {e}\n"
            "Fix: ensure all-MiniLM-L6-v2 is cached locally (no download needed normally).",
            title="Graceful stop", border_style="red"))
        sys.exit(0)

    try:
        llm_ok = bool(get_llm(temperature=0.0))
    except Exception as e:
        console.print(Panel(
            f"[red]Could not create an LLM:[/red] {e}\n"
            "Fix: set YOLO_AUTO_API_KEY (or OPENAI_API_KEY) in .env, or start Ollama.",
            title="Graceful stop", border_style="red"))
        sys.exit(0)
    del llm_ok

    ec = EmbedCache(embedder)
    examples = load_eval_set(N_QUESTIONS)
    console.print(f"Loaded [green]{len(examples)}[/green] question(s)\n")

    console.print("Building in-memory Chroma corpus…")
    vectorstore, corpus_texts = build_corpus()
    console.print(f"[green]✓[/green] {len(corpus_texts)} chunks embedded")

    gen = make_generator()
    hybrid = HybridRetriever(vectorstore, corpus_texts, ec)

    results = []
    for i, ex in enumerate(examples, 1):
        q = ex["question"]
        console.print(f"\n[bold cyan]{i}.[/bold cyan] {q}")
        row = {"question": q}
        for label, retrieve in (("A", lambda qq=q: retrieve_dense(vectorstore, qq)),
                                ("B", lambda qq=q: hybrid.retrieve(qq))):
            ctxs = retrieve()
            try:
                answer = gen(q, ctxs)
            except Exception as e:
                console.print(f"   [red]pipeline {label}: generation failed ({e}) — skipped[/red]")
                continue
            s = score_example(q, answer, ctxs, ex["ground_truth"], ec)
            row.update({f"{name}_{label}": s[name] for name in METRIC_NAMES})
            console.print(f"   [{label}] " + "  ".join(f"{n}={s[n]:.3f}" for n in METRIC_NAMES))
            console.print(f"       [dim]{answer[:120]}{'…' if len(answer) > 120 else ''}[/dim]")
        results.append(row)

    if not results:
        console.print(Panel("[red]No pipeline runs completed.[/red]", border_style="red"))
        sys.exit(0)

    # ---- Per-question table -------------------------------------------------
    t = Table(title=f"Per-question scores (A = dense, B = hybrid+rerank)", box=box.SIMPLE_HEAVY)
    t.add_column("#", justify="right", style="cyan")
    t.add_column("Question", max_width=36)
    for name in METRIC_NAMES:
        t.add_column(f"{name[:9]} A", justify="right")
        t.add_column(f"{name[:9]} B", justify="right")
    for i, r in enumerate(results, 1):
        cells = []
        for name in METRIC_NAMES:
            cells += [f"{r[f'{name}_A']:.3f}", f"{r[f'{name}_B']:.3f}"]
        t.add_row(str(i), r["question"][:36], *cells)
    console.print(t)

    # ---- Mean table + verdict ----------------------------------------------
    m = Table(title="Mean score per metric + verdict", box=box.ROUNDED)
    m.add_column("Metric", style="bold")
    m.add_column("A (dense)", justify="right")
    m.add_column("B (hybrid+rerank)", justify="right")
    m.add_column("Winner", justify="center")
    winners = {"A": 0, "B": 0, "tie": 0}
    for name in METRIC_NAMES:
        a = sum(r[f"{name}_A"] for r in results if f"{name}_A" in r) / max(1, len(results))
        b = sum(r[f"{name}_B"] for r in results if f"{name}_B" in r) / max(1, len(results))
        if abs(a - b) < 0.01:
            verdict, key = "tie", "tie"
        elif b > a:
            verdict, key = "B wins", "B"
        else:
            verdict, key = "A wins", "A"
        winners[key] += 1
        style = "green" if key != "A" else ("yellow" if key == "A" else "dim")
        m.add_row(name, f"{a:.3f}", f"{b:.3f}", f"[{style}]{verdict}[/{style}]")
    console.print(m)

    overall = max(winners, key=lambda k: winners[k])
    console.print(Panel.fit(
        f"B won {winners['B']}/{len(METRIC_NAMES)} metrics, A won {winners['A']}, "
        f"{winners['tie']} tie(s).\n"
        f"[bold]Verdict:[/bold] {'pipeline B (hybrid + RRF + rerank)' if overall == 'B' else 'pipeline A (naive dense)' if overall == 'A' else 'no clear winner'} "
        f"performs better on this small eval set.\n"
        "[dim]Caveat: 5 questions is a smoke test, not statistical proof — grow the golden set "
        "before making architecture decisions.[/dim]",
        title="🏁 Verdict", border_style="green"))


if __name__ == "__main__":
    main()
