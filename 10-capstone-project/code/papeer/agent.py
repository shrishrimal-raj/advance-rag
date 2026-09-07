"""Phase 3 — Agentic RAG workflow with LangGraph.

Graph:   retrieve -> grade -> generate -> reflect
         grade 'irrelevant' or reflect 'retry' loops back to retrieve (bounded).

Why an agent (not a straight pipeline)? A research assistant should *check its work*:
grade the retrieved context before spending LLM tokens on generation, and reflect on the
draft answer to catch weak/unsupported responses. Bounding to MAX_ITERATIONS prevents
runaway cost/latency — a hard production requirement.

Each node records its wall-clock time into state['stages'] so the CLI can show where time goes.

Run:  uv run python 10-capstone-project/code/papeer/agent.py "your question"
"""
from __future__ import annotations

import sys
import time
import pathlib
from typing import TypedDict

_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from rich.console import Console
from rich.table import Table

from papeer.config import MAX_ITERATIONS, get_llm
from papeer.retrieval import retrieve

# Force UTF-8 output so emoji/box-drawing render correctly on Windows consoles (cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

console = Console()


class AgentState(TypedDict, total=False):
    question: str
    context: str          # formatted, citation-tagged context block
    answer: str
    grade: str            # "relevant" | "irrelevant"
    reflect: str          # "ok" | "retry"
    iteration: int
    stages: list          # [(stage_name, seconds), ...]


# --------------------------------------------------------------------------- #
# Prompt templates (kept explicit & simple for a research assistant)
# --------------------------------------------------------------------------- #
GRADE_PROMPT = """You are grading whether the retrieved context is sufficient to answer a question.
Question: {question}

Context:
{context}

Respond with exactly one word: "relevant" if the context can support an answer, else "irrelevant"."""

GENERATE_PROMPT = """Answer the question using ONLY the provided context. Cite sources inline as [n].
If the context is insufficient, say so plainly. Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""

REFLECT_PROMPT = """Check this draft answer against the question and context. Is it complete and
faithful to the context (no fabrication)? Respond with exactly one word: "ok" or "retry".

Question: {question}
Context:
{context}
Draft answer: {answer}"""


def _format_context(hits: list[dict]) -> str:
    """Turn retrieval hits into a citation-tagged block: [1] (source) text ..."""
    if not hits:
        return "(no relevant passages found)"
    parts = []
    for h in hits:
        parts.append(f"[{h['rank']}] (source: {h['source']})\n{h['text']}")
    return "\n\n".join(parts)


def _llm_text(prompt: str) -> str:
    llm = get_llm(temperature=0.0)
    out = llm.invoke(prompt)
    return out.content if hasattr(out, "content") else str(out)


# --------------------------------------------------------------------------- #
# Graph nodes
# --------------------------------------------------------------------------- #
def node_retrieve(state: AgentState) -> dict:
    t0 = time.perf_counter()
    hits = retrieve(state["question"])
    ctx = _format_context(hits)
    dt = time.perf_counter() - t0
    return {"context": ctx, "stages": state.get("stages", []) + [("retrieve", dt)]}


def node_grade(state: AgentState) -> dict:
    t0 = time.perf_counter()
    raw = _llm_text(GRADE_PROMPT.format(question=state["question"], context=state["context"]))
    low = raw.strip().lower()
    grade = "irrelevant" if "irrelevant" in low else "relevant"
    dt = time.perf_counter() - t0
    return {"grade": grade, "stages": state.get("stages", []) + [("grade", dt)]}


def node_generate(state: AgentState) -> dict:
    t0 = time.perf_counter()
    ans = _llm_text(GENERATE_PROMPT.format(context=state["context"], question=state["question"]))
    dt = time.perf_counter() - t0
    return {"answer": ans.strip(), "stages": state.get("stages", []) + [("generate", dt)]}


def node_reflect(state: AgentState) -> dict:
    t0 = time.perf_counter()
    raw = _llm_text(REFLECT_PROMPT.format(
        question=state["question"], context=state["context"], answer=state.get("answer", "")
    ))
    decision = "retry" if "retry" in raw.strip().lower() else "ok"
    dt = time.perf_counter() - t0
    return {"reflect": decision,
            "iteration": state.get("iteration", 0) + 1,
            "stages": state.get("stages", []) + [("reflect", dt)]}


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #
def after_grade(state: AgentState) -> str:
    return "generate" if state.get("grade") == "relevant" else "reflect"


def after_reflect(state: AgentState) -> str:
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "end"
    return "retrieve" if state.get("reflect") == "retry" else "end"


def build_graph():
    from langgraph.graph import StateGraph, START, END
    g = StateGraph(AgentState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("grade", node_grade)
    g.add_node("generate", node_generate)
    g.add_node("reflect", node_reflect)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "reflect": "reflect"})
    g.add_edge("generate", "reflect")
    g.add_conditional_edges("reflect", after_reflect, {"retrieve": "retrieve", "end": END})
    return g.compile()


def run_agent(question: str) -> dict:
    """Run the full agent for one question. Returns {answer, context, stages, iteration}."""
    graph = build_graph()
    final = graph.invoke({
        "question": question, "context": "", "answer": "",
        "grade": "", "reflect": "", "iteration": 0, "stages": [],
    })
    return {
        "answer": final.get("answer", ""),
        "context": final.get("context", ""),
        "stages": final.get("stages", []),
        "iteration": final.get("iteration", 0),
    }


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Summarize Acme Robotics' support policy."
    res = run_agent(q)
    console.rule(f"❓ {q}")
    console.print(res["answer"])
    table = Table(title="⏱️ Stage timings")
    table.add_column("Stage")
    table.add_column("Seconds", justify="right")
    total = 0.0
    for name, secs in res["stages"]:
        table.add_row(name, f"{secs:.3f}")
        total += secs
    table.add_row("[bold]total[/bold]", f"[bold]{total:.3f}[/bold]")
    console.print(table)
    console.print(f"Iterations used: {res['iteration']}")
