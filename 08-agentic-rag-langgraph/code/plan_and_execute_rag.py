"""
Module 08 · Implementation C — Agentic RAG: PLAN-AND-EXECUTE (explicit StateGraph)
===================================================================================
The classic industry pattern for *complex* questions that a single retrieve→generate
pass can't handle:

    START → planner ──► executor ──┬─(steps left)──► executor …
                                   └─(all done)────► synthesizer → END

- **planner**   — one LLM call decomposes the question into 2–4 concrete steps,
                  each tagged `needs_retrieval` (RAG search) or not (pure reasoning).
- **executor**  — runs ONE step per invocation: optional vector search + one LLM call,
                  appending its result to the shared state (reducer = operator.add).
- **synthesizer**— merges all step results into the final cited answer.

Why a graph instead of a for-loop? The plan lives in *typed shared state*, so you can
add re-planning edges, human-in-the-loop interrupts, checkpointing, and parallel
executors later without rewriting control flow.

Run from project root:
    uv run python 08-agentic-rag-langgraph/code/plan_and_execute_rag.py

LLM: whatever shared.config.get_llm() resolves to (OpenAI / Yolo-Auto / Ollama).
Embeddings: local MiniLM (cached). If the LLM is unreachable you get a friendly hint
and exit 0 — never a stack trace.
"""
import sys
import pathlib
import json
import re
import operator
from typing import TypedDict, Annotated, List, Dict

# --- Course boilerplate: make `shared` importable from any cwd -------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

# --- Force UTF-8 output (Windows legacy consoles default to cp1252 and would
#     otherwise crash on arrows/emoji like ▶ → ✅). Never let a glyph kill the run.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.config import get_llm, get_embeddings, get_llm_provider_name
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END

console = Console()
ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "data" / "samples"


# --------------------------------------------------------------------------- #
# Vector store over data/samples/ (in-memory Chroma, always fresh)
# --------------------------------------------------------------------------- #
def build_vector_store():
    docs = []
    for p in sorted(SAMPLES_DIR.iterdir()):
        if p.is_file():
            docs.append(Document(page_content=p.read_text(encoding="utf-8", errors="ignore"),
                                 metadata={"source": p.name}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = [c for d in docs for c in splitter.split_documents([d])]
    console.print(f"[dim]Indexed {len(chunks)} chunks from {len(docs)} files into Chroma…[/dim]")
    return Chroma.from_documents(chunks, get_embeddings(), collection_name="plan_and_execute_rag")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _fmt_docs(docs):
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )


def _parse_json(text):
    """Tolerantly extract the first JSON object from an LLM reply (strips code fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end + 1])


def _llm_call(llm, system, prompt, label):
    """Run one LLM call; on failure print a friendly hint and return None."""
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return content.strip()
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]{label} failed:[/red] {e}")
        console.print("[yellow]Check your network / API key (see .env). Nothing to fix locally.[/yellow]")
        return None


def _get_llm_safe():
    """Build the LLM, turning a missing-backend ImportError into a precise, actionable hint."""
    try:
        return get_llm(temperature=0.0)
    except ImportError as e:
        console.print(f"[red]Missing LLM backend:[/red] {e}")
        console.print("[yellow]Set OPENAI_API_KEY or YOLO_AUTO_API_KEY in .env, or install Ollama.[/yellow]")
        sys.exit(0)


# --------------------------------------------------------------------------- #
# Typed state — the single source of truth every node reads/writes
# --------------------------------------------------------------------------- #
class PlanState(TypedDict):
    question: str
    plan: List[Dict]                                  # [{"task": str, "needs_retrieval": bool}]
    step_index: int                                   # which step the executor is on
    step_results: Annotated[List[str], operator.add]  # reducer APPENDS -> accumulates across steps
    final_answer: str


def main():
    console.rule("[bold cyan]Module 08 · C — Plan-and-Execute RAG (LangGraph)[/bold cyan]")
    console.print(f"[dim]LLM provider: {get_llm_provider_name()}[/dim]")
    vs = build_vector_store()
    llm = _get_llm_safe()

    def log(step, msg):
        console.print(f"[bold cyan]▶ {step}[/bold cyan]  [dim]{msg}[/dim]")

    # ---- Node: planner ------------------------------------------------------ #
    def planner(state):
        log("planner", f"decomposing: '{state['question']}'")
        prompt = (
            "You are a planning agent. Decompose the QUESTION into 2-4 concrete sub-steps. "
            "Each step must be independently answerable. Mark needs_retrieval=true when the "
            "step needs facts from the knowledge base, false for pure reasoning/computation.\n"
            f"QUESTION: {state['question']}\n\n"
            'Return ONLY JSON: {"steps": [{"task": "<sub-step>", "needs_retrieval": true|false}]}'
        )
        raw = _llm_call(llm, "You are a precise planning agent.", prompt, "planner")
        if raw is None:
            return {"plan": [], "step_index": 0}
        try:
            data = _parse_json(raw)
            steps = []
            for s in data.get("steps", [])[:4]:
                task = str(s.get("task", "")).strip()
                if task:
                    steps.append({"task": task, "needs_retrieval": bool(s.get("needs_retrieval", True))})
            if not steps:
                raise ValueError("empty steps list")
        except Exception as e:  # noqa: BLE001
            log("planner", f"could not parse plan JSON ({e}) → falling back to single-step plan")
            steps = [{"task": state["question"], "needs_retrieval": True}]
        return {"plan": steps, "step_index": 0}

    # ---- Node: executor (runs ONE step per invocation) ---------------------- #
    def executor(state):
        i = state["step_index"]
        step = state["plan"][i]
        tag = "search" if step["needs_retrieval"] else "reason"
        log(f"executor[{i + 1}/{len(state['plan'])}]", f"[{tag}] {step['task']}")

        context = "(no retrieval for this step)"
        if step["needs_retrieval"]:
            docs = vs.similarity_search(step["task"], k=3)
            sources = [d.metadata.get("source", "?") for d in docs]
            log(f"executor[{i + 1}/{len(state['plan'])}]", f"retrieved top-3: {sources}")
            context = _fmt_docs(docs)

        prior = "\n".join(state.get("step_results", [])) or "(no completed steps yet)"
        prompt = (
            "You are one step of a plan-and-execute pipeline. Answer ONLY this sub-step, "
            "concisely (2-4 sentences), using the CONTEXT and any COMPLETED STEPS. Cite "
            "sources inline as [n]. If the context is insufficient, say so.\n"
            f"COMPLETED STEPS:\n{prior}\n\nCONTEXT:\n{context}\n\nSUB-STEP: {step['task']}\n\nRESULT:"
        )
        result = _llm_call(llm, "You are a precise step executor.", prompt, f"executor[{i + 1}]")
        if result is None:
            result = f"(step {i + 1} unavailable — LLM error)"
        log(f"executor[{i + 1}/{len(state['plan'])}]", f"done ({len(result)} chars)")
        return {"step_index": i + 1,
                "step_results": [f"Step {i + 1} [{tag}]: {result}"]}

    # ---- Node: synthesizer --------------------------------------------------- #
    def synthesizer(state):
        log("synthesizer", f"merging {len(state['step_results'])} step results")
        results_block = "\n\n".join(state["step_results"])
        prompt = (
            "You are the final synthesizer of a plan-and-execute pipeline. Merge the STEP "
            "RESULTS into one coherent, complete answer to the original QUESTION. Keep "
            "inline citations. Do not invent facts beyond the step results.\n"
            f"QUESTION: {state['question']}\n\nSTEP RESULTS:\n{results_block}\n\nFINAL ANSWER:"
        )
        ans = _llm_call(llm, "You are a precise synthesis writer.", prompt, "synthesizer")
        if ans is None:
            ans = ("(synthesis unavailable — here are the raw step results)\n" + results_block)
        log("synthesizer", f"produced {len(ans)} chars")
        return {"final_answer": ans}

    # ---- Routing -------------------------------------------------------------- #
    def route_after_planner(state):
        if not state.get("plan"):
            log("route", "planner produced no plan → END (degraded)")
            return END
        return "executor"

    def route_after_executor(state):
        if state["step_index"] < len(state["plan"]):
            return "executor"
        return "synthesize"

    # ---- Build & compile the graph -------------------------------------------- #
    g = StateGraph(PlanState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("synthesize", synthesizer)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner, ["executor", END])
    g.add_conditional_edges("executor", route_after_executor, ["executor", "synthesize"])
    g.add_edge("synthesize", END)
    app = g.compile()

    # ---- Run -------------------------------------------------------------------- #
    question = ("How much more does the Quantum Laptop Pro 16 cost than the Pulse Smart Monitor 27, "
                "and what percentage more is that? Also, what do our notes say about HNSW indexing?")
    console.print(Panel.fit(f"[bold green]Question:[/bold green] {question}", border_style="green"))

    try:
        final_state = app.invoke({"question": question, "step_index": 0,
                                  "step_results": [], "plan": [], "final_answer": ""})
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Graph run failed:[/red] {e}")
        console.print("[yellow]Check your network / API key (see .env).[/yellow]")
        sys.exit(0)

    # ---- Pretty report ------------------------------------------------------------ #
    plan = final_state.get("plan", [])
    if not plan:
        console.print("[yellow]Degraded run: the planner could not produce a plan "
                      "(LLM unavailable or unparseable output). Nothing to execute.[/yellow]")
        console.rule()
        return
    t = Table(title="[bold]The Plan (as produced by the planner node)[/bold]", show_lines=True)
    t.add_column("#", justify="right")
    t.add_column("Sub-step")
    t.add_column("Mode")
    for i, s in enumerate(plan, 1):
        t.add_row(str(i), s["task"], "[green]RAG search[/green]" if s["needs_retrieval"] else "[yellow]reasoning[/yellow]")
    console.print(t)

    console.print(Panel.fit(final_state.get("final_answer", ""),
                            title="[bold blue]Final Answer (synthesizer)[/bold blue]", border_style="blue"))
    console.rule()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:  # noqa: BLE001 - last-resort friendly message, never a raw traceback
        console.print(f"[red]Something went wrong:[/red] {e}")
        console.print("[yellow]Check .env (API keys) and network connectivity.[/yellow]")
        sys.exit(0)
