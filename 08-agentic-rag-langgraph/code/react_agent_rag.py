"""
Module 08 · Implementation A — Agentic RAG: RAG AS A TOOL (ReAct agent)
=======================================================================
The LLM decides WHEN and HOW to retrieve. Retrieval is exposed as a *tool*
(`search_knowledge_base`) that a prebuilt LangGraph ReAct agent may call, alongside a
second tool (`calculator`) to demonstrate multi-tool use.

Run from project root:
    uv run python 08-agentic-rag-langgraph/code/react_agent_rag.py

No API keys required: local Ollama (llama3.1) + local sentence-transformers embeddings.
If Ollama isn't running you'll get a friendly hint instead of a stack trace.
"""
import sys
import pathlib
import re

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
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.prebuilt import create_react_agent

console = Console()
ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "data" / "samples"


# --------------------------------------------------------------------------- #
# Vector store over data/samples/ (in-memory Chroma, always fresh)
# --------------------------------------------------------------------------- #
def build_vector_store():
    """Read every file in data/samples/, chunk it, load into an in-memory Chroma."""
    docs = []
    for p in sorted(SAMPLES_DIR.iterdir()):
        if p.is_file():
            docs.append(Document(page_content=p.read_text(encoding="utf-8", errors="ignore"),
                                 metadata={"source": p.name}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = [c for d in docs for c in splitter.split_documents([d])]
    console.print(f"[dim]Indexed {len(chunks)} chunks from {len(docs)} files into Chroma…[/dim]")
    return Chroma.from_documents(chunks, get_embeddings(), collection_name="agentic_rag_tool")


# --------------------------------------------------------------------------- #
# Safe calculator helper (never eval raw model output)
# --------------------------------------------------------------------------- #
def _safe_eval(expression: str):
    expr = expression.strip()
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        raise ValueError("Only numbers and + - * / ( ) are allowed.")
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - input is whitelisted above


def _text(content) -> str:
    """Coerce message content (str or list of blocks) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


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


def main():
    console.rule("[bold cyan]Module 08 · A — ReAct Agent with RAG as a Tool[/bold cyan]")
    vs = build_vector_store()

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the knowledge base for facts about RAG, vector databases/HNSW,
        Acme Robotics products, and the product catalog. Returns the top-3 most
        relevant passages, each tagged with its source filename."""
        hits = vs.similarity_search(query, k=3)
        if not hits:
            return "No results found."
        return "\n\n".join(
            f"[{i}] (source: {h.metadata.get('source', '?')})\n{h.page_content.strip()}"
            for i, h in enumerate(hits, 1)
        )

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression such as '1899.99 + 3*399.00' or '(42*3)/7'.
        Supports + - * / and parentheses. Use it whenever the answer needs arithmetic."""
        try:
            value = _safe_eval(expression)
            return f"{expression} = {value}"
        except Exception as e:  # noqa: BLE001 - report back to the agent, don't crash
            return f"Could not evaluate '{expression}': {e}"

    llm = _get_llm_safe()
    agent = create_react_agent(llm, tools=[search_knowledge_base, calculator])

    system_prompt = (
        "You are a helpful research assistant with access to a knowledge base and a "
        "calculator. Use search_knowledge_base to look up facts BEFORE answering. Use "
        "calculator for any arithmetic. Cite the source filename when you quote a fact. "
        "Be concise."
    )

    questions = [
        "What is the payload capacity of the Falcon AMR, and how many hours of battery does it have?",
        "How much would it cost to buy one Quantum Laptop Pro 16 and three Orbit Ergonomic Chairs?",
    ]

    for q in questions:
        console.print(Panel.fit(f"[bold green]Q:[/bold green] {q}", border_style="green"))
        try:
            result = agent.invoke({"messages": [("system", system_prompt), ("user", q)]})
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Agent failed:[/red] {e}")
            console.print("[yellow]Is Ollama running? Try: ollama serve  (and: ollama pull llama3.1)[/yellow]")
            continue
        _print_trace(result["messages"])
    console.rule()


def _print_trace(messages):
    """Pretty-print the agent's journey: tool calls, tool results, and the final answer."""
    for m in messages:
        role = m.type
        if role == "human":
            continue  # already printed above
        if role == "ai":
            thought = _text(m.content).strip()
            if getattr(m, "tool_calls", None):
                if thought:
                    console.print(f"[bold magenta]🤖 thought:[/bold magenta] {thought}")
                for tc in m.tool_calls:
                    args = tc.get("args", {})
                    console.print(f"[bold magenta]🔧 tool_call:[/bold magenta] {tc['name']}({args})")
            else:
                console.print(Panel.fit(_text(m.content),
                                        title="[bold blue]✅ Final answer[/bold blue]",
                                        border_style="blue"))
        elif role == "tool":
            body = _text(m.content)
            if len(body) > 400:
                body = body[:400] + " …(truncated)"
            console.print(f"[bold yellow]↩ tool_result ({m.name}):[/bold yellow]\n[dim]{body}[/dim]")


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
