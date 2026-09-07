"""Evaluate a RAG pipeline with RAGAS (targeting ragas >= 0.2).

Loads the golden dataset, builds a Chroma-over-samples retriever + an LLM
generator, then runs BOTH:
  (1) individual metric scores on 2 examples (faithfulness, answer relevancy,
      context precision, context recall), and
  (2) a full ragas.evaluate() over the whole dataset with a summary table.

Works with NO API keys (local Ollama + local embeddings). Judge calls are SLOW
on local LLMs — see the warning printed at startup.

Run from the project root:
    uv run python 09-rag-evaluation-ragas/code/evaluate_pipeline.py

Env knobs:
    EVAL_LIMIT=N        cap the full evaluate() to the first N examples (default: all)
    INDIVIDUAL_EXAMPLES=N  how many examples to score in Part 1 (default: 2)
"""
import sys, pathlib, json, asyncio, os

# --- Course convention: make the project root importable --------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm, get_embeddings

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]     # 09-rag-evaluation-ragas
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]   # advance-rag
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
DATASET_PATH = MODULE_DIR / "data" / "eval_dataset.json"

TOP_K = 3
INDIVIDUAL_EXAMPLES = int(os.getenv("INDIVIDUAL_EXAMPLES", "2"))
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0")) or None       # 0/None -> all examples


def load_dataset():
    if not DATASET_PATH.exists():
        console.print(Panel(
            "[red]Golden dataset not found.[/red]\n"
            f"Expected: {DATASET_PATH}\n\n"
            "Build it first:\n"
            "  uv run python 09-rag-evaluation-ragas/code/build_dataset.py",
            title="Missing dataset", border_style="red"))
        sys.exit(1)
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Step 1: build the corpus (load + chunk + embed -> in-memory Chroma)         #
# --------------------------------------------------------------------------- #
def build_vectorstore():
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    try:
        from langchain_chroma import Chroma
    except ImportError:                       # older installs ship it here
        from langchain_community.vectorstores import Chroma

    docs = []
    for f in sorted(SAMPLES_DIR.iterdir()):
        if f.is_file():
            docs.append(Document(page_content=f.read_text(encoding="utf-8"),
                                 metadata={"source": f.name}))
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs)
    # In-memory (no persist_directory) => reproducible, no disk side-effects.
    emb = get_embeddings()
    # NOTE: pass embeddings POSITIONALLY. Some langchain-community versions raise
    # "got multiple values for keyword argument 'embedding_function'" when passed by name.
    try:
        return Chroma.from_documents(chunks, emb)
    except TypeError:
        return Chroma.from_documents(chunks, embedding_function=emb)


# --------------------------------------------------------------------------- #
# Step 2: retriever + generator callables (RAGAS-compatible)                  #
# --------------------------------------------------------------------------- #
PROMPT = """Answer the question using ONLY the provided context.
If the answer is not in the context, say exactly: "I don't know."
Be concise and factual.

Context:
{context}

Question: {question}
Answer:"""


def make_retriever(vectorstore):
    def retriever(question: str):
        """Retrieval stage: question -> top-k context strings."""
        return [d.page_content for d in vectorstore.similarity_search(question, k=TOP_K)]
    return retriever


def make_generator():
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm(temperature=0.0)

    def generator(question: str, contexts):
        """Generation stage: question + contexts -> answer string."""
        resp = chain.invoke({"context": "\n\n".join(contexts), "question": question})
        return resp.content
    return generator


# --------------------------------------------------------------------------- #
# Step 3: RAGAS setup                                                         #
# --------------------------------------------------------------------------- #
def setup_ragas():
    # NOTE: RAGAS APIs change between versions. This targets ragas >= 0.2.
    # If your imports differ, check:
    #   uv run python -c "import ragas; print(ragas.__version__)"
    # Older 0.1.x used `from ragas import Metrics` and `Metrics.Faithfulness()`.
    from ragas import evaluate
    try:
        from ragas import Metrics  # present in some versions; harmless if absent
    except Exception:
        Metrics = None
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_llm = LangchainLLMWrapper(get_llm())
    embedder = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
    }
    for m in metrics.values():
        m.llm = judge_llm
    metrics["answer_relevancy"].embeddings = embedder   # AR needs an embedding model

    return evaluate, SingleTurnSample, EvaluationDataset, metrics, judge_llm, embedder


def score_single(metric, sample):
    """Score one sample with one metric (ragas >= 0.2 async API, w/ fallback)."""
    if hasattr(metric, "single_turn_asynccreate"):
        return asyncio.run(metric.single_turn_asynccreate(sample, None))
    if hasattr(metric, "score"):
        return metric.score(sample)
    raise AttributeError("No scoring method found on this RAGAS metric version.")


def fmt(v):
    """Format a possibly-NaN metric value for display."""
    try:
        import pandas as pd
        if pd.isna(v):
            return "—"
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def main():
    console.print(Panel.fit(
        "[bold]RAGAS pipeline evaluation[/bold]\n"
        "Judge calls are [yellow]slow on local LLMs[/yellow]. To speed up learning, set "
        "[cyan]EVAL_LIMIT=3[/cyan] (caps the full run) or add OPENAI_API_KEY to .env.",
        title="⏱️  Heads-up", border_style="yellow"))

    dataset = load_dataset()
    console.print(f"Loaded [green]{len(dataset)}[/green] golden example(s) from {DATASET_PATH.name}\n")

    vectorstore = build_vectorstore()
    retriever = make_retriever(vectorstore)
    generator = make_generator()
    console.print("[green]✓[/green] Built in-memory Chroma corpus + retriever/generator\n")

    evaluate, SingleTurnSample, EvaluationDataset, metrics, judge_llm, embedder = setup_ragas()
    console.print("[green]✓[/green] RAGAS ready — metrics: " + ", ".join(metrics) + "\n")

    def make_sample(ex):
        ctxs = retriever(ex["question"])
        return SingleTurnSample(
            user_input=ex["question"],
            retrieved_contexts=ctxs,
            reference=ex["ground_truth"],
            response=generator(ex["question"], ctxs),
        )

    # ---- Part 1: individual metric scores on N examples -------------------- #
    console.rule("[bold]Part 1 — Individual metric scores[/bold]")
    try:
        for ex in dataset[:INDIVIDUAL_EXAMPLES]:
            s = make_sample(ex)
            console.print(f"\n[bold cyan]Q:[/bold cyan] {ex['question']}")
            ans = s.response or ""
            console.print(f"[dim]retrieved {len(s.retrieved_contexts)} context(s); "
                          f"answer: {ans[:160]}{'…' if len(ans) > 160 else ''}[/dim]")
            for name, m in metrics.items():
                try:
                    val = float(score_single(m, s))
                    console.print(f"  • {name:<18} {val:.3f}")
                except Exception as e:
                    console.print(f"  • {name:<18} [red]error: {e}[/red]")
    except Exception as e:
        console.print(Panel(f"[red]Individual-metric run failed:[/red] {e}\n"
                            "Check Ollama is running (ollama serve) or set OPENAI_API_KEY.",
                            title="Error", border_style="red"))

    # ---- Part 2: full ragas.evaluate() over the dataset -------------------- #
    console.rule("[bold]Part 2 — Full ragas.evaluate()[/bold]")
    examples = dataset if EVAL_LIMIT is None else dataset[:EVAL_LIMIT]
    console.print(f"Evaluating [bold]{len(examples)}[/bold] example(s) × {len(metrics)} metric(s)… "
                  "(this is the slow part on local LLMs)\n")
    try:
        ragas_ds = EvaluationDataset([make_sample(ex) for ex in examples])
        try:
            result = evaluate(dataset=ragas_ds, metrics=list(metrics.values()),
                              llm=judge_llm, embeddings=embedder)
        except TypeError:
            # Some versions don't take llm/embeddings at top level; metrics already have them.
            result = evaluate(dataset=ragas_ds, metrics=list(metrics.values()))
        df = result.to_pandas()

        t = Table(title="Per-example scores", box=box.SIMPLE_HEAVY)
        t.add_column("#", justify="right", style="cyan")
        t.add_column("Question", max_width=40)
        for name in metrics:
            t.add_column(name, justify="right")
        for i, (_, r) in enumerate(df.iterrows(), 1):
            q = r.get("user_input", "")
            q = q if isinstance(q, str) else str(q)
            t.add_row(str(i), q[:40], *[fmt(r.get(name)) for name in metrics])
        console.print(t)

        m = Table(title="Mean score summary (higher is better)", box=box.ROUNDED)
        m.add_column("Metric", style="bold")
        m.add_column("Mean", justify="right")
        for name in metrics:
            vals = [float(x) for x in df[name].dropna()]
            mean = sum(vals) / len(vals) if vals else float("nan")
            m.add_row(name, fmt(mean))
        console.print(m)
        console.print("\n[dim]Interpretation guide & decision tree: see 02-learning.md §9.[/dim]")
    except Exception as e:
        console.print(Panel(f"[red]Full evaluation failed:[/red] {e}\n"
                            "Common causes: Ollama not running, RAGAS version mismatch, or timeout. "
                            "Try EVAL_LIMIT=2 or set OPENAI_API_KEY.",
                            title="Error", border_style="red"))


if __name__ == "__main__":
    main()
