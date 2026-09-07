"""
Module 08 · Implementation B — Agentic RAG: EXPLICIT STATEGRAPH (industry pattern)
==================================================================================
A hand-built LangGraph StateGraph where each RAG stage is a NODE and the LLM drives the
CONTROL FLOW via CONDITIONAL EDGES:

    START → retrieve → grade_documents ─┬─(relevant)──────────────► generate → reflect ─┬─(SUPPORTED)→ END
                                        ├─(none & retries left)───► rewrite_query ─► retrieve
                                        └─(retries exhausted)─────► generate ("not enough info")
                                                                                          └─(UNSUPPORTED & iter<max)→ retrieve

Compiled with a MemorySaver checkpointer so a multi-turn conversation keeps memory.

Run from project root:
    uv run python 08-agentic-rag-langgraph/code/agentic_graph_rag.py

No API keys required: local Ollama (llama3.1) + local sentence-transformers embeddings.
"""
import sys
import pathlib
import json
import re
import operator
from typing import TypedDict, Annotated, List, Optional

# --- Course boilerplate: make `shared` importable from any cwd -------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

# --- Force UTF-8 output (Windows legacy consoles default to cp1252 and would
#     otherwise crash on arrows/emoji like ▶ → ✅). Never let a glyph kill the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - some streams aren't reconfigurable; ignore
        pass

from rich.console import Console
from rich.panel import Panel

from shared.config import get_llm, get_embeddings
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

console = Console()
ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "data" / "samples"
MAX_ITERATIONS = 3  # hard cap on how many times we (re-)retrieve per question


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
    return Chroma.from_documents(chunks, get_embeddings(), collection_name="agentic_rag_graph")


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


def _llm_call(llm, prompt, label):
    """Run an LLM call; on failure print a friendly Ollama hint and return None."""
    try:
        return llm.invoke([SystemMessage(content="You are a precise RAG pipeline component."),
                           HumanMessage(content=prompt)])
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]{label} failed:[/red] {e}")
        console.print("[yellow]Is Ollama running? Try: ollama serve  (and: ollama pull llama3.1)[/yellow]")
        return None


def _get_llm_safe():
    """Build the LLM, turning a missing-backend ImportError into a precise, actionable hint."""
    try:
        return get_llm(temperature=0.0)
    except ImportError as e:
        console.print(f"[red]Missing LLM backend:[/red] {e}")
        console.print(
            "[yellow]The no-API-key path uses Ollama. Install its integration once, then start it:\n"
            "    uv add langchain-ollama\n"
            "    ollama serve   &   ollama pull llama3.1[/yellow]"
        )
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# Typed state
# --------------------------------------------------------------------------- #
class State(TypedDict):
    question: str
    chat_history: Annotated[List[str], operator.add]   # reducer APPENDS across turns -> memory
    retrieved_docs: List[Document]
    graded_relevance: bool
    answer: str
    iteration: int                                       # safety counter (incremented in retrieve)
    reflection_supported: bool
    reflection_critique: str
    rewritten_query: Optional[str]


def main():
    console.rule("[bold cyan]Module 08 · B — Explicit StateGraph Agentic RAG[/bold cyan]")
    vs = build_vector_store()
    llm = _get_llm_safe()
    journey: List[str] = []  # per-turn list of nodes that ran (for the trace summary)

    def log(step, msg):
        console.print(f"[bold cyan]▶ {step}[/bold cyan]  [dim]{msg}[/dim]")

    # ---- Nodes ------------------------------------------------------------- #
    def retrieve(state):
        journey.append("retrieve")
        q = state.get("rewritten_query") or state["question"]
        log("retrieve", f"query='{q}'  (iteration {state['iteration']} → {state['iteration'] + 1})")
        docs = vs.similarity_search(q, k=4)
        log("retrieve", f"top-4 sources: {[d.metadata.get('source', '?') for d in docs]}")
        return {"retrieved_docs": docs, "iteration": state["iteration"] + 1}

    def grade_documents(state):
        journey.append("grade_documents")
        docs = state["retrieved_docs"]
        if not docs:
            log("grade_documents", "no docs to grade")
            return {"graded_relevance": False}
        prompt = (
            "You are a strict relevance grader. Decide which numbered documents are actually "
            f"needed to answer the QUESTION.\nQUESTION: {state['question']}\nDOCUMENTS:\n{_fmt_docs(docs)}\n\n"
            'Return ONLY JSON: {"relevant": [<1-based indices>], "reason": "<short>"}'
        )
        resp = _llm_call(llm, prompt, "grade_documents")
        relevant_idx, reason = [], ""
        if resp is not None:
            try:
                data = _parse_json(resp.content)
                relevant_idx = [int(i) for i in data.get("relevant", []) if 1 <= int(i) <= len(docs)]
                reason = data.get("reason", "")
            except Exception as e:  # noqa: BLE001
                log("grade_documents", f"could not parse grader JSON ({e}); keeping all docs")
                relevant_idx = list(range(1, len(docs) + 1))
        kept = [docs[i - 1] for i in relevant_idx]
        log("grade_documents", f"kept {len(kept)}/{len(docs)} relevant — {reason}")
        return {"retrieved_docs": kept, "graded_relevance": len(kept) > 0}

    def rewrite_query(state):
        journey.append("rewrite_query")
        prompt = (
            "The previous retrieval found nothing relevant. Rewrite the question as a sharper "
            f"search query.\nOriginal: {state['question']}\nReturn ONLY the rewritten query text "
            "(one line, no quotes)."
        )
        resp = _llm_call(llm, prompt, "rewrite_query")
        new_q = state["question"]
        if resp is not None and resp.content.strip():
            new_q = resp.content.strip().strip('"').splitlines()[0]
        log("rewrite_query", f"new query='{new_q}'")
        return {"rewritten_query": new_q}

    def generate(state):
        journey.append("generate")
        docs = state["retrieved_docs"]
        if not docs:
            ans = "I don't have enough information in the knowledge base to answer that confidently."
            log("generate", "no relevant docs → 'not enough info' answer (no LLM call)")
            return {"answer": ans, "chat_history": [f"Assistant: {ans}"]}
        history_block = "\n".join(state.get("chat_history", [])) or "(no prior turns)"
        prompt = (
            "Answer the question using ONLY the provided context. Cite sources inline as [n]. "
            "If the context is insufficient, say you don't have enough information.\n"
            f"CONVERSATION SO FAR:\n{history_block}\n\nCONTEXT:\n{_fmt_docs(docs)}\n\n"
            f"QUESTION: {state['question']}\n\nANSWER:"
        )
        resp = _llm_call(llm, prompt, "generate")
        ans = resp.content.strip() if resp else "(generation unavailable — see error above)"
        log("generate", f"produced {len(ans)} chars")
        return {"answer": ans, "chat_history": [f"Assistant: {ans}"]}

    def reflect(state):
        journey.append("reflect")
        docs = state["retrieved_docs"]
        if not docs:
            log("reflect", "nothing to verify (no docs) — accepting 'not enough info'")
            return {"reflection_supported": True, "reflection_critique": "no docs; answered honestly"}
        prompt = (
            "Check whether the ANSWER is fully supported by the CONTEXT (no fabrication).\n"
            f"CONTEXT:\n{_fmt_docs(docs)}\nANSWER: {state['answer']}\n\n"
            'Return ONLY JSON: {"supported": true|false, "critique": "<why>", '
            '"refined_query": "<a better search query, or null>"}'
        )
        resp = _llm_call(llm, prompt, "reflect")
        supported, critique, refined = False, "", None
        if resp is not None:
            try:
                data = _parse_json(resp.content)
                supported = bool(data.get("supported", False))
                critique = data.get("critique", "")
                refined = data.get("refined_query")
            except Exception as e:  # noqa: BLE001
                log("reflect", f"could not parse reflection JSON ({e}); assuming unsupported")
        verdict = "SUPPORTED ✅" if supported else "UNSUPPORTED ❌"
        log("reflect", f"{verdict} — {critique}")
        update = {"reflection_supported": supported, "reflection_critique": critique}
        if refined:
            update["rewritten_query"] = refined
        return update

    # ---- Conditional edges (routing functions) ---------------------------- #
    def route_after_grade(state):
        if state["graded_relevance"]:
            return "generate"
        if state["iteration"] < MAX_ITERATIONS:
            log("route", "no relevant docs → rewrite_query (retry retrieval)")
            return "rewrite_query"
        log("route", "no relevant docs & retries exhausted → generate ('not enough info')")
        return "generate"

    def route_after_reflect(state):
        if state["reflection_supported"]:
            log("route", "answer supported → END")
            return END
        if state["iteration"] < MAX_ITERATIONS:
            log("route", "answer unsupported → re-retrieve (loop back)")
            return "retrieve"
        log("route", "answer unsupported & iterations exhausted → END")
        return END

    # ---- Build & compile the graph --------------------------------------- #
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("reflect", reflect)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grade, ["generate", "rewrite_query"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "reflect")
    g.add_conditional_edges("reflect", route_after_reflect, ["retrieve", END])

    app = g.compile(checkpointer=MemorySaver())

    # ---- Run a 2-turn conversation (demonstrates checkpointing/memory) ---- #
    turns = [
        "Explain how HNSW indexing works and name its key parameters.",
        "Now, which Acme Robotics product has the larger payload, and what is its warranty?",
    ]
    config = {"configurable": {"thread_id": "demo-conversation"}}

    for i, q in enumerate(turns, 1):
        console.print(Panel.fit(f"[bold green]Turn {i} · User:[/bold green] {q}", border_style="green"))
        journey.clear()
        initial = {
            "question": q,
            "iteration": 0,
            "rewritten_query": None,
            "graded_relevance": False,
            "reflection_supported": False,
            "answer": "",
            "chat_history": [f"User: {q}"],
        }
        try:
            app.invoke(initial, config)  # nodes print their own trace as they run
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Graph run failed:[/red] {e}")
            console.print("[yellow]Is Ollama running? Try: ollama serve  (and: ollama pull llama3.1)[/yellow]")
            break

        final_state = app.get_state(config).values
        answer = final_state.get("answer", "")
        console.print(Panel.fit(answer, title=f"[bold blue]Turn {i} · Answer[/bold blue]", border_style="blue"))
        console.print(f"[dim]Journey: {' → '.join(journey)}[/dim]")
        console.print(f"[dim]Memory (chat_history): {final_state.get('chat_history', [])}[/dim]\n")

    console.rule()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:  # noqa: BLE001 - last-resort friendly message, never a raw traceback
        console.print(f"[red]Something went wrong:[/red] {e}")
        console.print(
            "[yellow]Tips: ensure Ollama is running (`ollama serve`, then `ollama pull llama3.1`). "
            "The first run also downloads the local embedding model (~90 MB) — check your network.[/yellow]"
        )
