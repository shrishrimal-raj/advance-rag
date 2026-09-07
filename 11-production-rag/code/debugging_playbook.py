"""Failure triage playbook for production RAG (Module 11) — offline, no LLM needed.

A scripted simulation of 5 classic production failures. For each one you get:
    SYMPTOMS        — what users / dashboards see
    DIAGNOSTICS     — the exact steps to isolate the cause (with simulated evidence)
    ROOT CAUSE      — what actually went wrong
    FIX             — the concrete remediation + prevention

Interactive: pick a case from the menu (rich Prompt).
Non-interactive:  uv run python 11-production-rag/code/debugging_playbook.py --demo

Run:  uv run python 11-production-rag/code/debugging_playbook.py [--demo]
"""
from __future__ import annotations

import sys
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))  # -> project root

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# --------------------------------------------------------------------------- #
# The five cases: symptoms -> diagnostics -> root cause -> fix
# --------------------------------------------------------------------------- #
CASES: list[dict] = [
    {
        "id": 1,
        "title": "Empty retrieval (0 results)",
        "symptoms": [
            "Answer is always 'I don't know' or generic filler",
            "Retrieval stage returns [] in traces; latency is suspiciously fast",
            "Happens for ALL queries, not just hard ones",
        ],
        "diagnostics": [
            ("Check collection count", "col.count() == 0 → index is empty or you're querying the wrong collection name"),
            ("Check embedding provider match", "query embeddings from model A vs index built with model B → distances are garbage"),
            ("Check filter expression", "a metadata filter like {'status': 'published'} matching nothing silently returns 0 rows"),
            ("Check distance threshold", "an over-strict max_distance prunes everything"),
        ],
        "root_cause": "Metadata filter `{'classification': 'public'}` was added after re-indexing, but the new "
                      "ingestion job writes `classification: 'Public'` (capital P). Filter matches zero chunks.",
        "fix": "Normalize metadata values at ingestion time (lowercase enum). Add a startup assertion: "
               "`assert col.count() > 0` and a canary query in CI that must return ≥1 result.",
    },
    {
        "id": 2,
        "title": "Low-confidence retrieval scores",
        "symptoms": [
            "Answers are technically on-topic but miss the specific fact asked about",
            "Top-k cosine similarities cluster around 0.30–0.45 (healthy: 0.55+)",
            "RAGAS context_recall trending down week over week",
        ],
        "diagnostics": [
            ("Inspect top-k passages", "are they the RIGHT document, wrong passage? → chunking problem"),
            ("Compare query vs passage length", "long chatty queries dilute the embedding; try a step-back/condensed query"),
            ("A/B dense-only vs hybrid", "if BM25 finds it and dense doesn't, add hybrid + RRF"),
            ("Check for domain drift", "new document types (tables, code) embed poorly with a prose-tuned model"),
        ],
        "root_cause": "Chunks were grown from 300 to 1200 chars to 'give the LLM more context'. Big chunks embed as "
                      "blended topics, so similarity to any single question drops.",
        "fix": "Shrink chunks (~300–500 chars) and use parent-document retrieval: retrieve small child chunks, "
               "pass the parent to the LLM. Add a min-similarity gate that triggers a fallback (BM25 or 'I don't know').",
    },
    {
        "id": 3,
        "title": "Hallucinated answer (confidently wrong)",
        "symptoms": [
            "Answer cites sources [1][2] but the cited passages don't contain the claim",
            "RAGAS faithfulness < 0.6 while answer_relevancy looks fine",
            "Users report 'made-up numbers'",
        ],
        "diagnostics": [
            ("Replay the trace", "pull the exact retrieved passages for the failing request_id"),
            ("Ask: is the fact IN the context?", "if NO → hallucination; if YES but misstated → generation/prompt problem"),
            ("Check prompt grounding", "does the system prompt say 'answer ONLY from context, else say I don't know'?"),
            ("Check temperature & model", "temp > 0 adds invention risk; weak models ignore grounding instructions"),
        ],
        "root_cause": "Retrieval returned low-relevance passages (see case 2), and the prompt lacked a refusal "
                      "instruction, so the LLM filled the gap from parametric memory.",
        "fix": "Two-layer fix: (1) gate retrieval — if best score < threshold, refuse instead of generating; "
               "(2) harden the prompt ('If the context does not contain the answer, respond exactly: I don't know') "
               "+ output guardrail that flags answers contradicting the context.",
    },
    {
        "id": 4,
        "title": "LLM timeout under load",
        "symptoms": [
            "p95 latency spikes from 3 s to 30 s+ during traffic peaks",
            "Error rate climbs; some requests hang until client timeout",
            "Provider dashboard shows 429s / queueing",
        ],
        "diagnostics": [
            ("Split latency by stage", "embed/retrieve fine, generate dominates → it's the LLM call"),
            ("Count concurrent requests", "compare against provider rate limits (RPM/TPM)"),
            ("Check prompt size trend", "context grew (bigger k, bigger chunks) → slower + more expensive per call"),
            ("Check retry policy", "retries without backoff amplify load and make timeouts worse"),
        ],
        "root_cause": "No request cap: 40 concurrent users × unbounded retries hit the provider's TPM limit; "
                      "queued requests time out at 30 s.",
        "fix": "Add a concurrency semaphore + exponential-backoff retries with jitter, a hard per-request timeout "
               "(e.g. 15 s) with graceful fallback message, and route easy queries to a smaller/faster model. "
               "Cache hot queries so peaks don't all reach the LLM.",
    },
    {
        "id": 5,
        "title": "Embedding dimension mismatch",
        "symptoms": [
            "Hard error at query time: 'Expected embedding of length 384, got 768' (or silent NaN distances)",
            "Started right after a dependency/model upgrade",
            "Index works in old code, breaks in new code",
        ],
        "diagnostics": [
            ("Print both dims", "len(indexed_vec) vs len(query_vec) — different models have different dims"),
            ("Check .env / config", "EMBEDDING_MODEL changed between index build and query time"),
            ("Check library version", "a sentence-transformers upgrade may have swapped the default model"),
            ("Check normalization", "same dims but different spaces (model A index vs model B query) also breaks cosine"),
        ],
        "root_cause": "The index was built with MiniLM-L6-v2 (384-d); a config change switched the query-time model "
                      "to a 768-d variant. Vectors live in incompatible spaces.",
        "fix": "Treat the embedding model as part of the index schema: store model name + dim in collection metadata, "
               "assert equality at startup, and REBUILD the index whenever the model changes (you cannot mix spaces).",
    },
]


def show_case(case: dict) -> None:
    console.print(Panel(f"[bold red]Case {case['id']}:[/bold red] {case['title']}", border_style="red"))

    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_column("Section", style="bold cyan", width=14)
    t.add_column("Detail")
    t.add_row("SYMPTOMS", "\n".join("• " + s for s in case["symptoms"]))
    t.add_row("DIAGNOSTICS", "\n".join(f"• {step}: {detail}" for step, detail in case["diagnostics"]))
    t.add_row("ROOT CAUSE", case["root_cause"])
    t.add_row("FIX", case["fix"])
    console.print(t)
    console.print()


def main() -> None:
    demo = "--demo" in sys.argv
    console.rule("🔧 RAG Failure Triage Playbook")

    if demo:
        console.print("[dim]--demo: running all five cases non-interactively.[/dim]\n")
        for c in CASES:
            show_case(c)
        return

    menu = Table(title="Pick a failure to triage")
    menu.add_column("#", justify="right")
    menu.add_column("Failure")
    for c in CASES:
        menu.add_row(str(c["id"]), c["title"])
    menu.add_row("0", "[dim]Exit[/dim]")
    console.print(menu)

    while True:
        try:
            choice = int(input("Case number (0 to quit): ").strip())
        except (EOFError, KeyboardInterrupt):
            break
        if choice == 0:
            break
        if 1 <= choice <= len(CASES):
            show_case(CASES[choice - 1])
        else:
            console.print("[red]Invalid choice.[/red]")
    console.print("[green]✔ Playbook complete — remember: trace first, guess never.[/green]")


if __name__ == "__main__":
    main()
