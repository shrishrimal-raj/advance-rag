"""RAG Evaluation Harness CLI.

Usage:
  uv run python cli.py run --pipeline A --dataset evals/dataset.jsonl --report out/report.md
  uv run python cli.py run --pipeline B ...          # adds RAGAS (needs LLM key)
  uv run python cli.py gate --baseline evals/baseline.json --tolerance 0.05
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from harness.config import get_embedder, get_llm, get_llm_provider_name
from harness.metrics import evaluate_question, ragas_scores

console = Console()
METRIC_KEYS = ["context_keyword_coverage", "answer_relevancy", "faithfulness_proxy"]


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_pipeline(pipeline: str, dataset: list[dict]) -> dict:
    embed = get_embedder(prefer_local_model=True)
    llm = get_llm() if pipeline.upper() == "B" else None
    results = []
    for i, row in enumerate(dataset, 1):
        with console.status(f"Evaluating {i}/{len(dataset)} ..."):
            scores = evaluate_question(row["question"], row["contexts"], row["answer"], embed)
        if llm is not None:
            ragas = ragas_scores(row["question"], row.get("reference_answer", ""), row["contexts"], row["answer"], llm)
            scores["ragas_available"] = bool(ragas)
            scores.update({f"ragas_{k}": v for k, v in ragas.items()})
        results.append({"question": row["question"], **scores})
    averages = {k: round(sum(r[k] for r in results) / len(results), 4) for k in METRIC_KEYS}
    return {"pipeline": pipeline.upper(), "n": len(results), "averages": averages, "results": results}


def write_report(out: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# RAG Evaluation Report — pipeline {out['pipeline']}", "",
             f"Questions: {out['n']}", "", "| Metric | Average |", "|---|---|"]
    for k, v in out["averages"].items():
        lines.append(f"| {k} | {v:.4f} |")
    lines += ["", "## Per-question", ""]
    for r in out["results"]:
        lines.append(f"- **{r['question'][:70]}** — " + ", ".join(f"{k}={r[k]:.3f}" for k in METRIC_KEYS))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(out: dict) -> None:
    t = Table(title=f"Pipeline {out['pipeline']} — {out['n']} questions")
    t.add_column("Metric")
    t.add_column("Average", justify="right")
    for k, v in out["averages"].items():
        t.add_row(k, f"{v:.4f}")
    console.print(t)


def cmd_run(args) -> int:
    dataset = load_dataset(Path(args.dataset))
    if not dataset:
        console.print(f"[red]No rows in {args.dataset}[/red]")
        return 1
    out = run_pipeline(args.pipeline, dataset)
    print_table(out)
    if args.report:
        write_report(out, Path(args.report))
        console.print(f"Report: [green]{args.report}[/green]")
    results_path = Path(args.results or f"out/results_{args.pipeline.upper()}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    console.print(f"Results: [green]{results_path}[/green]")
    return 0


def cmd_gate(args) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    base_avg = baseline.get("averages", baseline)
    dataset = load_dataset(Path(args.dataset))
    out = run_pipeline(args.pipeline, dataset)
    tol = args.tolerance
    drops = []
    for k in METRIC_KEYS:
        if k in base_avg:
            delta = out["averages"][k] - base_avg[k]
            if delta < -tol:
                drops.append((k, base_avg[k], out["averages"][k], delta))
    if drops:
        for k, b, c, d in drops:
            console.print(f"[red]GATE FAIL[/red] {k}: {b:.4f} -> {c:.4f} (drop {-d:.4f} > tol {tol})")
        return 1
    console.print(f"[green]GATE PASS[/green] no metric dropped more than {tol}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="RAG evaluation & MLOps harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="evaluate a dataset")
    pr.add_argument("--pipeline", choices=["A", "B"], default="A",
                    help="A=offline metrics only, B=+RAGAS (needs LLM key)")
    pr.add_argument("--dataset", default="evals/dataset.jsonl")
    pr.add_argument("--report", default=None)
    pr.add_argument("--results", default=None)
    pr.set_defaults(fn=cmd_run)
    pg = sub.add_parser("gate", help="compare against a baseline")
    pg.add_argument("--baseline", required=True)
    pg.add_argument("--dataset", default="evals/dataset.jsonl")
    pg.add_argument("--pipeline", choices=["A", "B"], default="A")
    pg.add_argument("--tolerance", type=float, default=0.05)
    pg.set_defaults(fn=cmd_gate)
    args = p.parse_args()
    try:
        return args.fn(args)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
