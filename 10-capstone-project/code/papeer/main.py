"""Phase 5 — CLI demo: run 3 questions through the FULL Papeer pipeline.

Flow:
    1. Ensure the index exists (build it via ingestion if missing).
    2. For each of 3 questions, run the LangGraph agent (retrieve -> grade -> generate -> reflect).
    3. Print the answer, per-stage timings, and total latency for each question.

This is the "does it all work end-to-end" check for the capstone.

Run:  uv run python 10-capstone-project/code/papeer/main.py
"""
from __future__ import annotations

import sys
import time
import pathlib

_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from rich.console import Console
from rich.table import Table

from papeer.config import CHUNKS_JSON
from papeer.ingestion import build_index
from papeer.agent import run_agent

# Force UTF-8 output so emoji/box-drawing render correctly on Windows consoles (cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

console = Console()

DEMO_QUESTIONS = [
    "What navigation technology does the Falcon AMR use?",
    "Summarize Acme Robotics' support and warranty policy.",
    "Which vector index builds a multi-layer graph, and what are its key parameters?",
]


def ensure_index() -> None:
    if not CHUNKS_JSON.exists():
        console.print("[cyan]Index not found — building it now…[/cyan]")
        build_index(verbose=True)
    else:
        console.print("[green]✔ Index present — skipping ingestion.[/green]")


def main() -> None:
    console.rule("📄 Papeer — Capstone Research Assistant")
    t_start = time.perf_counter()
    ensure_index()

    summary_rows = []
    for i, q in enumerate(DEMO_QUESTIONS, start=1):
        console.rule(f"Question {i}/{len(DEMO_QUESTIONS)}")
        console.print(f"[bold cyan]❓ {q}[/bold cyan]")

        t0 = time.perf_counter()
        try:
            res = run_agent(q)
        except Exception as e:  # noqa: BLE001 - surface a clean error, don't crash the demo
            console.print(f"[red]✖ Pipeline error: {e}[/red]")
            console.print("[dim](Is Ollama running? `ollama serve` + `ollama pull llama3.1`, "
                          "or set OPENAI_API_KEY in .env)[/dim]")
            continue
        total = time.perf_counter() - t0

        console.print("\n[bold green]🤖 Answer:[/bold green]\n")
        console.print(res["answer"])

        table = Table(title="⏱️ Stage timings", show_lines=False)
        table.add_column("Stage")
        table.add_column("Seconds", justify="right")
        stage_sum = 0.0
        for name, secs in res["stages"]:
            table.add_row(name, f"{secs:.3f}")
            stage_sum += secs
        table.add_row("[bold]total (wall)[/bold]", f"[bold]{total:.3f}[/bold]")
        console.print(table)
        console.print(f"Iterations used: {res['iteration']}  |  stage-sum: {stage_sum:.3f}s\n")
        summary_rows.append((q[:40], f"{stage_sum:.2f}", f"{total:.2f}"))

    # Final roll-up table across all questions.
    roll = Table(title="🏁 Run summary")
    roll.add_column("Question", max_width=42)
    roll.add_column("Stage-sum (s)", justify="right")
    roll.add_column("Wall (s)", justify="right")
    for row in summary_rows:
        roll.add_row(*row)
    console.print(roll)
    console.print(f"\n[bold]Full demo completed in {time.perf_counter() - t_start:.1f}s.[/bold]")


if __name__ == "__main__":
    main()
