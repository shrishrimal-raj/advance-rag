"""Approach 2 — RAG evaluation WITHOUT RAGAS: custom embedding-based metrics.

Implements the four core RAGAS-style metrics from scratch, using only local
MiniLM embeddings (no LLM, no API keys, fully offline):

  (a) Faithfulness proxy   : claim-level support — split the answer into
                              claims (sentences); a claim is "supported" if its
                              max cosine similarity to any retrieved context
                              >= threshold. Score = supported / total claims.
  (b) Context precision    : mean semantic overlap between each retrieved
                              context and the reference answer (are the
                              retrieved chunks actually on-topic?).
      Context recall       : fact coverage — split the reference answer into
                              facts (clauses); a fact is "covered" if some
                              retrieved context is similar enough.
                              Score = covered / total facts.
  (c) Answer relevancy     : cosine similarity(question, answer).

Eval set: the golden dataset built by build_dataset.py (10 Q / reference /
context triples derived from data/samples/). If the JSON is missing it is
rebuilt on the fly. Because no LLM is involved, the *reference answer itself*
is scored as the "generated answer" — this measures how well the metric
machinery works on known-good data (a sanity baseline: scores should be high).

Run from the project root:
    uv run python 09-rag-evaluation-ragas/code/approach_2_custom_metrics.py

Env knobs:
    EVAL_LIMIT=N   cap the number of examples (default: all)
"""
import sys, pathlib, json, re, os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- Course convention: make the project root importable --------------------
MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]     # 09-rag-evaluation-ragas
PROJECT_ROOT = MODULE_DIR.parent                             # advance-rag
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))  # sibling scripts

from shared.config import get_embeddings  # noqa: E402
from rich.console import Console          # noqa: E402
from rich.table import Table              # noqa: E402
from rich.panel import Panel              # noqa: E402
from rich import box                      # noqa: E402

console = Console()

DATASET_PATH = MODULE_DIR / "data" / "eval_dataset.json"
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0")) or None

# --- Tunables (MiniLM cosine similarities typically live in 0.1 .. 0.7) ------
CLAIM_SUPPORT_THRESHOLD = 0.30   # (a) min sim for a claim to count as supported
FACT_COVERAGE_THRESHOLD = 0.30   # (b) min sim for a reference fact to count as covered


# --------------------------------------------------------------------------- #
# Embedding helpers                                                           #
# --------------------------------------------------------------------------- #
class EmbedCache:
    """Thin memoizing wrapper so identical texts are embedded once."""

    def __init__(self, embedder):
        import numpy as np
        self._np = np
        self._embedder = embedder
        self._cache: dict[str, object] = {}

    def vec(self, text: str):
        key = text.strip()
        if key not in self._cache:
            self._cache[key] = self._np.asarray(self._embedder.embed_query(key), dtype="float32")
        return self._cache[key]

    @staticmethod
    def cosine(a, b) -> float:
        denom = float((a @ b) / (max(float(a @ a) ** 0.5, 1e-9) * max(float(b @ b) ** 0.5, 1e-9)))
        return denom


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _facts(text: str) -> list[str]:
    """Finer-grained than sentences: also split on commas/semicolons."""
    out = []
    for s in _sentences(text):
        out.extend(f.strip(" .;:") for f in re.split(r"[;,]", s) if f.strip(" .;:"))
    return out


# --------------------------------------------------------------------------- #
# The three custom metrics                                                    #
# --------------------------------------------------------------------------- #
def faithfulness_proxy(answer: str, contexts: list[str], ec: EmbedCache):
    """(a) Fraction of answer claims whose best context similarity >= threshold.

    Returns (score, per-claim detail list of (claim, best_sim, supported)).
    """
    claims = _sentences(answer)
    if not claims or not contexts:
        return 0.0, []
    ctx_vecs = [ec.vec(c) for c in contexts]
    detail = []
    supported = 0
    for claim in claims:
        cv = ec.vec(claim)
        best = max(ec.cosine(cv, cv2) for cv2 in ctx_vecs)
        ok = best >= CLAIM_SUPPORT_THRESHOLD
        supported += ok
        detail.append((claim, best, ok))
    return supported / len(claims), detail


def context_precision(contexts: list[str], reference: str, ec: EmbedCache) -> float:
    """(b) Mean semantic overlap between retrieved contexts and the reference.

    High => every retrieved chunk is on-topic. Low => retrieval pulled noise.
    """
    if not contexts:
        return 0.0
    ref_v = ec.vec(reference)
    sims = [max(0.0, ec.cosine(ec.vec(c), ref_v)) for c in contexts]
    return sum(sims) / len(sims)


def context_recall(contexts: list[str], reference: str, ec: EmbedCache) -> float:
    """(b) Fraction of reference facts covered by at least one retrieved context."""
    facts = _facts(reference)
    if not facts or not contexts:
        return 0.0
    ctx_vecs = [ec.vec(c) for c in contexts]
    covered = 0
    for fact in facts:
        fv = ec.vec(fact)
        if max(ec.cosine(fv, cv) for cv in ctx_vecs) >= FACT_COVERAGE_THRESHOLD:
            covered += 1
    return covered / len(facts)


def answer_relevancy(question: str, answer: str, ec: EmbedCache) -> float:
    """(c) Cosine similarity between the question and the answer."""
    if not question.strip() or not answer.strip():
        return 0.0
    return max(0.0, ec.cosine(ec.vec(question), ec.vec(answer)))


def score_example(question: str, answer: str, contexts: list[str],
                  reference: str, ec: EmbedCache) -> dict:
    """Score one example on all four custom metrics."""
    faith, _ = faithfulness_proxy(answer, contexts, ec)
    return {
        "question": question,
        "faithfulness": faith,
        "context_precision": context_precision(contexts, reference, ec),
        "context_recall": context_recall(contexts, reference, ec),
        "answer_relevancy": answer_relevancy(question, answer, ec),
    }


# --------------------------------------------------------------------------- #
# Eval set                                                                    #
# --------------------------------------------------------------------------- #
def load_eval_set(limit: int | None = None) -> list[dict]:
    """Load the golden dataset; rebuild it if the JSON is missing."""
    if DATASET_PATH.exists():
        data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    else:
        from build_dataset import build_dataset  # sibling script
        data = build_dataset()
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATASET_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data[:limit] if limit else data


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
METRIC_NAMES = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


def main():
    console.print(Panel.fit(
        "[bold]Custom RAG metrics — no RAGAS, no LLM[/bold]\n"
        "Faithfulness proxy · context precision/recall · answer relevancy\n"
        "All computed with local MiniLM embeddings (offline).",
        title="📏 Approach 2: Custom Metrics", border_style="cyan"))

    try:
        embedder = get_embeddings()
        _ = embedder.embed_query("warmup")  # force model load early
    except Exception as e:
        console.print(Panel(
            f"[red]Local embeddings unavailable:[/red] {e}\n\n"
            "Fix: make sure sentence-transformers can load "
            "all-MiniLM-L6-v2 (it should already be cached in %USERPROFILE%\\.cache\\huggingface).\n"
            "No network/API key is required for this script.",
            title="Graceful stop", border_style="red"))
        sys.exit(0)

    ec = EmbedCache(embedder)
    examples = load_eval_set(EVAL_LIMIT)
    console.print(f"Loaded [green]{len(examples)}[/green] example(s) from {DATASET_PATH.name}\n")

    rows = []
    for i, ex in enumerate(examples, 1):
        # No LLM here: score the reference answer itself as the "generated"
        # answer. On known-good data every metric should land high — that is
        # the sanity baseline proving the metric machinery works.
        s = score_example(ex["question"], ex["ground_truth"], ex["contexts"],
                          ex["ground_truth"], ec)
        rows.append(s)
        console.print(f"[bold cyan]{i}.[/bold cyan] {ex['question']}")
        for name in METRIC_NAMES:
            console.print(f"   • {name:<18} {s[name]:.3f}")
        console.print()

    t = Table(title="Per-example custom metrics (reference answer vs. its own contexts)",
              box=box.SIMPLE_HEAVY)
    t.add_column("#", justify="right", style="cyan")
    t.add_column("Question", max_width=42)
    for name in METRIC_NAMES:
        t.add_column(name, justify="right")
    for i, s in enumerate(rows, 1):
        t.add_row(str(i), s["question"][:42], *[f"{s[n]:.3f}" for n in METRIC_NAMES])
    console.print(t)

    m = Table(title="Mean scores (sanity baseline — expect high values)", box=box.ROUNDED)
    m.add_column("Metric", style="bold")
    m.add_column("Mean", justify="right")
    for name in METRIC_NAMES:
        vals = [r[name] for r in rows]
        m.add_row(name, f"{sum(vals) / len(vals):.3f}" if vals else "—")
    console.print(m)
    console.print("\n[dim]Limitations: embedding-similarity proxies are coarse — short facts "
                  "(e.g. country names) and JSON-shaped contexts can score low even when "
                  "correctly supported. That is exactly why RAGAS/LLM judges exist; use "
                  "these for fast offline triage, not final verdicts.[/dim]")
    console.print("[dim]Reuse these functions in ab_pipeline_comparison.py to score real "
                  "pipeline outputs.[/dim]")


if __name__ == "__main__":
    main()
