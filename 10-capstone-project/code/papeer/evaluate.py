"""Phase 4 — Evaluation-in-the-loop with RAGAS.

A small built-in GOLDEN SET (question + ground-truth answer) is run through the real
retrieval + generation pipeline, then scored with RAGAS:
    - faithfulness        : is the answer supported by the retrieved context?
    - answer_relevancy    : does the answer address the question?
    - context_precision   : are the relevant passages ranked near the top?
    - context_recall      : did retrieval surface the ground-truth-relevant content?

In production this runs in CI as a *gate*: block a model/index change if faithfulness drops
below threshold. Here we print a report and degrade gracefully if RAGAS isn't importable.

Run:  uv run python 10-capstone-project/code/papeer/evaluate.py
"""
from __future__ import annotations

import sys
import pathlib

_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from rich.console import Console
from rich.table import Table

from papeer.config import CHUNKS_JSON
from papeer.retrieval import retrieve
from papeer.agent import _format_context, _llm_text, GENERATE_PROMPT

# Force UTF-8 output so emoji/box-drawing render correctly on Windows consoles (cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

console = Console()

# --------------------------------------------------------------------------- #
# Built-in golden set (grounded in data/samples/)
# --------------------------------------------------------------------------- #
GOLDEN_SET = [
    {
        "question": "What navigation technology does the Falcon AMR use?",
        "ground_truth": "The Falcon AMR uses LiDAR + SLAM navigation.",
    },
    {
        "question": "How many years of warranty does Acme Robotics offer?",
        "ground_truth": "Acme Robotics offers a 3-year warranty.",
    },
    {
        "question": "Which index type builds a multi-layer graph for fast ANN search?",
        "ground_truth": "HNSW (Hierarchical Navigable Small World) builds a multi-layer graph.",
    },
    {
        "question": "What is the payload capacity of the Heron Arm X2?",
        "ground_truth": "The Heron Arm X2 has a payload capacity of 25 kg.",
    },
]


def _generate(question: str, context: str) -> str:
    return _llm_text(GENERATE_PROMPT.format(context=context, question=question)).strip()


def _build_rows() -> list[dict]:
    """Run each golden question through retrieve + generate to build RAGAS input rows."""
    rows = []
    for item in GOLDEN_SET:
        q = item["question"]
        hits = retrieve(q)
        ctx = _format_context(hits)
        answer = _generate(q, ctx)
        rows.append({
            "question": q,
            "ground_truth": item["ground_truth"],
            "contexts": [h["text"] for h in hits],
            "answer": answer,
        })
        console.print(f"[dim]evaluated: {q}[/dim]")
    return rows


def _manual_report(rows: list[dict]) -> None:
    """Fallback when RAGAS is unavailable: a simple lexical-overlap proxy."""
    table = Table(title="📊 Manual eval (RAGAS unavailable — lexical overlap proxy)")
    table.add_column("Question", max_width=45)
    table.add_column("Overlap", justify="right")
    for r in rows:
        gt_tokens = set(r["ground_truth"].lower().split())
        ans_tokens = set(r["answer"].lower().split())
        overlap = len(gt_tokens & ans_tokens) / max(1, len(gt_tokens))
        table.add_row(r["question"], f"{overlap:.2f}")
    console.print(table)
    console.print("[yellow]Install/upgrade ragas for full metrics.[/yellow]")


def run_evaluation() -> None:
    if not CHUNKS_JSON.exists():
        console.print("[red]Index not built — run ingestion first.[/red]")
        return

    rows = _build_rows()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
        )
        ds = Dataset.from_list(rows)
        result = evaluate(ds, metrics=[Faithfulness(), AnswerRelevancy(),
                                       ContextPrecision(), ContextRecall()])
        df = result.to_pandas()
        console.rule("📊 RAGAS Report")
        with console.capture() as cap:
            from rich import print as rprint
            rprint(df.to_string(index=False))
        console.print(cap.get())
        # Pull headline numbers if present.
        try:
            faith = float(df["faithfulness"].mean())
            console.print(f"[bold]Mean faithfulness: {faith:.3f}[/bold] "
                          f"({'✅ > 0.8' if faith > 0.8 else '⚠️ below 0.8 target'})")
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001 - graceful degradation
        console.print(f"[yellow]RAGAS evaluation failed ({e}); showing manual report.[/yellow]")
        _manual_report(rows)


if __name__ == "__main__":
    run_evaluation()
