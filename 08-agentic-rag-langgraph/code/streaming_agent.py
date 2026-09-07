"""
Module 08 · Implementation E — Agentic RAG: TOKEN STREAMING (astream_events)
============================================================================
A small LangGraph RAG agent (retrieve → generate) run with
`app.astream_events(version="v2")` so tokens print AS THEY ARRIVE, with stage banners
showing which graph node is active:

    [stage] retrieval … done in 0.4s
    [stage] generation
    <tokens stream here…>

Streaming detection & fallback: some providers don't emit token-level chunks. We count
`on_chat_model_stream` events; if zero arrive we print a note and show the complete
(non-streamed) answer from the final state instead. If the event stream itself errors,
we fall back to a plain `app.invoke`.

Run from project root:
    uv run python 08-agentic-rag-langgraph/code/streaming_agent.py

LLM: whatever shared.config.get_llm() resolves to (OpenAI / Yolo-Auto / Ollama).
Embeddings: local MiniLM (cached). If the LLM is unreachable you get a friendly hint
and exit 0 — never a stack trace.
"""
import sys
import pathlib
import asyncio
import time

# --- Course boilerplate: make `shared` importable from any cwd -------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

# --- Force UTF-8 output (Windows legacy consoles default to cp1252 and would
#     otherwise crash on arrows/emoji like ▶ → ✅). Never let a glyph kill the run.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel

from shared.config import get_llm, get_embeddings, get_llm_provider_name
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

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
    return Chroma.from_documents(chunks, get_embeddings(), collection_name="streaming_agent")


# --------------------------------------------------------------------------- #
# Small LangGraph RAG agent: retrieve → generate
# --------------------------------------------------------------------------- #
class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str


def build_agent(vs, llm):
    def retrieve(state):
        docs = vs.similarity_search(state["question"], k=4)
        console.print(f"[bold cyan][stage] retrieval[/bold cyan] "
                      f"[dim]top-4 sources: {[d.metadata.get('source', '?') for d in docs]}[/dim]")
        return {"retrieved_docs": docs}

    def generate(state):
        context = "\n\n".join(
            f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content.strip()}"
            for i, d in enumerate(state["retrieved_docs"], 1)
        ) or "(no documents retrieved)"
        resp = llm.invoke([
            SystemMessage(content="Answer using ONLY the provided context. Cite sources inline as [n]."),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {state['question']}\n\nANSWER:"),
        ])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"answer": content.strip()}

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def _get_llm_safe():
    """Build the LLM, turning a missing-backend ImportError into a precise, actionable hint."""
    try:
        return get_llm(temperature=0.0)
    except ImportError as e:
        console.print(f"[red]Missing LLM backend:[/red] {e}")
        console.print("[yellow]Set OPENAI_API_KEY or YOLO_AUTO_API_KEY in .env, or install Ollama.[/yellow]")
        sys.exit(0)


# --------------------------------------------------------------------------- #
# Streaming runner
# --------------------------------------------------------------------------- #
async def run_streaming(app, question):
    """Stream events; return (streamed_text, final_answer, streamed_any_token)."""
    streamed_parts: List[str] = []
    final_answer = ""
    t0 = time.perf_counter()
    first_token_at = None

    async for event in app.astream_events(
        {"question": question, "retrieved_docs": [], "answer": ""},
        config={"configurable": {}},
        version="v2",
    ):
        kind, name = event.get("event"), event.get("name", "")
        if kind == "on_chain_start" and name == "generate":
            console.print(f"[bold cyan][stage] generation[/bold cyan] "
                          f"[dim](retrieval took {time.perf_counter() - t0:.1f}s — now streaming)[/dim]")
        elif kind == "on_chat_model_stream":
            tok = event["data"].get("chunk")
            text = getattr(tok, "content", "")
            if isinstance(text, list):  # some providers send content blocks
                text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter() - t0
                    console.print(f"[dim]first token after {first_token_at:.1f}s:[/dim] ", end="")
                console.print(text, end="", highlight=False)
                streamed_parts.append(text)
        elif kind == "on_chain_end" and name == "generate":
            data = event.get("data", {}).get("output") or {}
            if isinstance(data, dict):
                final_answer = data.get("answer", final_answer)

    console.print()  # newline after the token stream
    return "".join(streamed_parts), final_answer, bool(streamed_parts)


async def main():
    console.rule("[bold cyan]Module 08 · E — Token-Streaming Agentic RAG[/bold cyan]")
    console.print(f"[dim]LLM provider: {get_llm_provider_name()}[/dim]")
    vs = build_vector_store()
    llm = _get_llm_safe()
    app = build_agent(vs, llm)

    question = "Explain how HNSW indexing works and name its key parameters."
    console.print(Panel.fit(f"[bold green]Question:[/bold green] {question}", border_style="green"))

    try:
        streamed, final_answer, had_tokens = await run_streaming(app, question)
    except Exception as e:  # noqa: BLE001 — event stream itself failed; fall back to invoke
        console.print(f"[yellow]Event streaming failed ({e}); falling back to non-streamed invoke.[/yellow]")
        try:
            state = app.invoke({"question": question, "retrieved_docs": [], "answer": ""})
            final_answer = state.get("answer", "")
            had_tokens = False
        except Exception as e2:  # noqa: BLE001
            console.print(f"[red]Non-streamed fallback also failed:[/red] {e2}")
            console.print("[yellow]Check your network / API key (see .env).[/yellow]")
            sys.exit(0)

    if not had_tokens:
        console.print("[yellow]Note: the provider did not emit token-level stream events, "
                      "so this is the complete (non-streamed) answer:[/yellow]\n")
        console.print(final_answer or "(no answer produced)")
    else:
        console.print(f"\n[dim]{len(streamed)} chars streamed token-by-token "
                      f"(final state answer = {len(final_answer or streamed)} chars).[/dim]")

    console.rule()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:  # noqa: BLE001 - last-resort friendly message, never a raw traceback
        console.print(f"[red]Something went wrong:[/red] {e}")
        console.print("[yellow]Check .env (API keys) and network connectivity.[/yellow]")
        sys.exit(0)
