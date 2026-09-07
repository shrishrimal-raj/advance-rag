"""Token & cost accounting for RAG (Module 11) — works offline, uses LLM if available.

Measures, for 3 sample queries:
    - retrieved-context size (chars -> estimated tokens)
    - prompt tokens (system + user + context)
    - completion tokens
using the provider's `response.usage` when available, else the char/4 heuristic.

Then prices the same workload under two provider placeholders:
    - OpenAI gpt-4o-mini   ($0.15 / $0.60 per 1M input/output tokens)
    - Yolo-Auto qwen3.8-27b ($0.30 / $1.20 per 1M input/output tokens)  [placeholders]

and quantifies the classic savings levers as % estimates:
    - smaller k (retrieve fewer passages)
    - smaller chunks (shorter passages)
    - caching (serve repeats without any LLM call)

Run:  uv run python 11-production-rag/code/cost_optimization.py
"""
from __future__ import annotations

import sys
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))  # -> project root

from rich.console import Console
from rich.table import Table

console = Console()

# --------------------------------------------------------------------------- #
# Pricing placeholders (per 1M tokens). Update from your provider's price page.
# --------------------------------------------------------------------------- #
PRICING = {
    "OpenAI gpt-4o-mini":     {"input": 0.15, "output": 0.60},
    "Yolo-Auto qwen3.8-27b":  {"input": 0.30, "output": 1.20},
}

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions strictly from the provided "
    "context. Cite sources as [n]. If the context does not contain the answer, say so."
)

SAMPLE_QUERIES = [
    "What problem does RAG solve compared to fine-tuning?",
    "How does RAG reduce hallucination?",
    "What advanced components do industry deployments add?",
]


def est_tokens(text: str) -> int:
    """char/4 heuristic — good enough for budgeting (~±15% on English prose)."""
    return max(1, len(text) // 4)


def load_passages() -> list[str]:
    """Chunk the sample doc into ~300-char passages (no splitter needed here)."""
    p = pathlib.Path(__file__).resolve().parents[2] / "data" / "samples" / "rag_overview.txt"
    text = p.read_text(encoding="utf-8").strip()
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 20]
    passages, buf = [], ""
    for s in sentences:
        if buf and len(buf) + len(s) > 300:
            passages.append(buf.strip())
            buf = ""
        buf += s + ". "
    if buf.strip():
        passages.append(buf.strip())
    return passages


def retrieve(passages: list[str], query: str, k: int = 3) -> list[str]:
    """Tiny keyword-overlap retriever so the demo runs with zero dependencies."""
    q_terms = {w for w in query.lower().replace("?", "").split() if len(w) > 3}
    scored = []
    for i, p in enumerate(passages):
        p_terms = {w for w in p.lower().split() if len(w) > 3}
        scored.append((len(q_terms & p_terms), i))
    scored.sort(reverse=True)
    return [passages[i] for _, i in scored[:k]]


def call_llm(llm, prompt: str) -> tuple[str, int | None, int | None]:
    """Return (answer, prompt_tokens|None, completion_tokens|None) using usage when present."""
    resp = llm.invoke(prompt)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    pt = ct = None
    usage = getattr(resp, "usage_metadata", None)
    if isinstance(usage, dict):
        pt, ct = usage.get("input_tokens"), usage.get("output_tokens")
    if pt is None:
        info = getattr(resp, "llm_output", None) or {}
        det = (info.get("token_usage") or info.get("usage") or {})
        pt = det.get("prompt_tokens")
        ct = det.get("completion_tokens")
    return content, pt, ct


def main() -> None:
    console.rule("💰 Cost Optimization Demo")

    # -- LLM availability (graceful degradation) ---------------------------- #
    llm = None
    try:
        from shared.config import get_llm, get_llm_provider_name
        llm = get_llm(temperature=0.0)
        console.print(f"[dim]LLM provider: {get_llm_provider_name()}[/dim]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]LLM unavailable ({e.__class__.__name__}) — "
                      f"running in HEURISTIC mode (no real generation; char/4 estimates only).[/yellow]")

    passages = load_passages()
    rows = []
    for q in SAMPLE_QUERIES:
        ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(retrieve(passages, q)))
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{ctx}\n\nQuestion: {q}"
        if llm is not None:
            try:
                answer, pt, ct = call_llm(llm, prompt)
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]LLM call failed ({e.__class__.__name__}) — "
                              f"falling back to heuristic for remaining queries.[/yellow]")
                llm = None
                answer, pt, ct = "(heuristic)", None, None
        else:
            answer, pt, ct = "(heuristic)", None, None
        rows.append({
            "query": q,
            "ctx_chars": len(ctx),
            "ctx_tok": est_tokens(ctx),
            "prompt_tok": pt if pt is not None else est_tokens(prompt),
            "comp_tok": ct if ct is not None else est_tokens(answer),
            "usage": pt is not None,
        })

    # -- Per-query token table ---------------------------------------------- #
    t1 = Table(title="Tokens per stage (3 sample queries)")
    t1.add_column("Query", max_width=42)
    t1.add_column("Context chars", justify="right")
    t1.add_column("Context tok (est)", justify="right")
    t1.add_column("Prompt tok", justify="right")
    t1.add_column("Completion tok", justify="right")
    t1.add_column("Source")
    for r in rows:
        src = "[green]usage[/green]" if r["usage"] else "[yellow]char/4[/yellow]"
        t1.add_row(r["query"], f"{r['ctx_chars']:,}", f"{r['ctx_tok']:,}",
                   f"{r['prompt_tok']:,}", f"{r['comp_tok']:,}", src)
    console.print(t1)

    # -- Cost table across providers ----------------------------------------- #
    tot_in = sum(r["prompt_tok"] for r in rows)
    tot_out = sum(r["comp_tok"] for r in rows)
    t2 = Table(title=f"Cost of this workload ({tot_in:,} in / {tot_out:,} out tokens)")
    t2.add_column("Provider")
    t2.add_column("Input $/1M", justify="right")
    t2.add_column("Output $/1M", justify="right")
    t2.add_column("Cost (this run)", justify="right")
    t2.add_column("Cost @ 10k req/day", justify="right")
    for name, p in PRICING.items():
        cost = tot_in * p["input"] / 1e6 + tot_out * p["output"] / 1e6
        t2.add_row(name, f"{p['input']:.2f}", f"{p['output']:.2f}",
                   f"${cost:.4f}", f"${cost * 10_000:,.2f}")
    console.print(t2)

    # -- Savings levers -------------------------------------------------------- #
    base_in = tot_in
    tot_ctx = sum(r["ctx_tok"] for r in rows)
    # Levers that shrink CONTEXT save only the context share of the prompt;
    # caching skips the LLM call entirely (input AND output).
    levers = [
        ("Smaller k (3 → 1 passage)", tot_ctx * (1 - 1 / 3) / base_in,
         "fewer retrieved passages in the prompt"),
        ("Smaller chunks (300 → 150 chars)", tot_ctx * 0.5 / base_in,
         "each passage carries less filler"),
        ("Caching (assume 40% hit rate)", 0.40,
         "hits skip the LLM entirely (in + out)"),
    ]
    t3 = Table(title="Savings levers (estimated % reduction vs baseline)")
    t3.add_column("Lever")
    t3.add_column("Est. saving", justify="right")
    t3.add_column("New input tokens", justify="right")
    t3.add_column("Why it works")
    for name, frac, why in levers:
        frac = min(frac, 1.0)
        new_in = int(base_in * (1 - frac))
        t3.add_row(name, f"{frac:.0%}", f"{new_in:,}", why)
    console.print(t3)
    console.print("\n[green]✔ Caching is the biggest lever: a hit costs $0.0000, not just less.[/green]")


if __name__ == "__main__":
    main()
