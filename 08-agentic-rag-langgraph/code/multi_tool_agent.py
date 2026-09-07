"""
Module 08 · Implementation D — Agentic RAG: MULTI-TOOL ReAct AGENT
===================================================================
A prebuilt LangGraph ReAct agent with THREE tools, and the LLM picks which one to
call at every step — no hard-coded routing:

    vector_search(query)            — semantic search over the whole knowledge base
    metadata_filter_search(source)  — fetch chunks by metadata (e.g. only products.csv)
    calculator(expression)         — safe arithmetic (whitelisted chars, no raw eval of prose)

The demo question deliberately needs ALL THREE tools, so the printed trace shows the
agent choosing a *different* tool for each sub-question.

Run from project root:
    uv run python 08-agentic-rag-langgraph/code/multi_tool_agent.py

LLM: whatever shared.config.get_llm() resolves to (OpenAI / Yolo-Auto / Ollama).
Embeddings: local MiniLM (cached). If the LLM is unreachable you get a friendly hint
and exit 0 — never a stack trace.
"""
import sys
import pathlib
import re

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
from langchain_core.messages import AIMessage, ToolMessage
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
    docs = []
    for p in sorted(SAMPLES_DIR.iterdir()):
        if p.is_file():
            docs.append(Document(page_content=p.read_text(encoding="utf-8", errors="ignore"),
                                 metadata={"source": p.name}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = [c for d in docs for c in splitter.split_documents([d])]
    console.print(f"[dim]Indexed {len(chunks)} chunks from {len(docs)} files into Chroma…[/dim]")
    return Chroma.from_documents(chunks, get_embeddings(), collection_name="multi_tool_agent")


# --------------------------------------------------------------------------- #
# The three tools
# --------------------------------------------------------------------------- #
def _safe_eval(expression: str):
    expr = expression.strip()
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        raise ValueError("Only numbers and + - * / ( ) are allowed.")
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - input is whitelisted above


def make_tools(vs):
    @tool
    def vector_search(query: str) -> str:
        """Semantic search over the whole knowledge base. Use for open-ended factual
        questions like 'what do the notes say about X?'. Returns the top matching
        chunks with their source file names."""
        docs = vs.similarity_search(query, k=3)
        if not docs:
            return "No documents found."
        return "\n\n".join(
            f"[source: {d.metadata.get('source', '?')}]\n{d.page_content.strip()}" for d in docs
        )

    @tool
    def metadata_filter_search(source: str) -> str:
        """Fetch knowledge-base chunks filtered BY METADATA — pass the exact source
        file name (e.g. 'products.csv' or 'company_profile.json'). Use when you need
        everything from one specific document rather than semantic matches."""
        res = vs.get(where={"source": source}, limit=5)
        docs = res.get("documents", [])
        metas = res.get("metadatas", []) or [{}] * len(docs)
        if not docs:
            return f"No chunks found with source='{source}'."
        return "\n\n".join(
            f"[source: {md.get('source', '?')}]\n{doc.strip()}" for doc, md in zip(docs, metas)
        )

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression (+ - * / parentheses), e.g.
        '(1600 - 1200) / 1200 * 100'. Use for any numeric comparison or percentage."""
        try:
            return str(_safe_eval(expression))
        except Exception as e:  # noqa: BLE001
            return f"Calculation error: {e}"

    return [vector_search, metadata_filter_search, calculator]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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
        console.print("[yellow]Set OPENAI_API_KEY or YOLO_AUTO_API_KEY in .env, or install Ollama.[/yellow]")
        sys.exit(0)


def print_trace(messages):
    """Print the full ReAct loop: every thought, tool call, and observation."""
    console.rule("[bold magenta]Full tool-call trace[/bold magenta]")
    step = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                step += 1
                args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                console.print(f"[bold cyan]Step {step} · tool call:[/bold cyan] "
                              f"[green]{tc['name']}[/green]({args})")
        elif isinstance(msg, ToolMessage):
            body = _text(msg.content).strip().replace("\n", " ")
            console.print(f"[dim]   ↳ observation ({msg.name}): {body[:220]}{'…' if len(body) > 220 else ''}[/dim]")
        elif isinstance(msg, AIMessage) and _text(msg.content).strip():
            console.print(f"[bold blue]Final answer:[/bold blue] {_text(msg.content).strip()}")


def main():
    console.rule("[bold cyan]Module 08 · D — Multi-tool ReAct Agent[/bold cyan]")
    console.print(f"[dim]LLM provider: {get_llm_provider_name()}[/dim]")
    vs = build_vector_store()
    llm = _get_llm_safe()

    tools = make_tools(vs)
    agent = create_react_agent(
        llm,
        tools,
        prompt=(
            "You are a research assistant with access to a small company knowledge base "
            "and a calculator. The knowledge base contains these source files: "
            "products.csv, company_profile.json, rag_overview.txt, vector_db_notes.md. "
            "Answer the user's question using your tools. When you need facts from a specific "
            "file, use metadata_filter_search; for general questions use vector_search; for "
            "arithmetic use calculator. Cite source file names."
        ),
    )

    question = ("How much more does the Quantum Laptop Pro 16 cost than the Pulse Smart Monitor 27, "
                "and what percentage more is that? What payload capacity and battery life does "
                "the Falcon AMR have? And what do our notes say about HNSW's key parameters?")
    console.print(Panel.fit(f"[bold green]Question:[/bold green] {question}", border_style="green"))

    try:
        result = agent.invoke({"messages": [("user", question)]},
                              config={"recursion_limit": 25})
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Agent run failed:[/red] {e}")
        console.print("[yellow]Check your network / API key (see .env).[/yellow]")
        sys.exit(0)

    messages = result.get("messages", [])
    print_trace(messages)

    used = {tc["name"] for m in messages if isinstance(m, AIMessage) for tc in getattr(m, "tool_calls", [])}
    console.print(f"\n[dim]Tools used this run: {sorted(used) or 'none'}[/dim]")
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
