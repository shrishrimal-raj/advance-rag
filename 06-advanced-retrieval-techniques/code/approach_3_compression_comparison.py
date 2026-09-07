"""
Module 6 · Approach 3 — Contextual Compression: ON vs OFF
=========================================================
Side-by-side comparison of the same retrieval pipeline with and without
LLM-based context compression:

  OFF: retrieve top-5 chunks -> feed ALL of them to the generator
  ON : retrieve top-5 chunks -> LLM keeps ONLY the sentences that answer
       the question -> feed the compressed context to the generator

The script prints a token-count delta table so you can see exactly how many
tokens compression saves per document and in total.

LLM-dependent: needs a reachable LLM (Ollama / OpenAI / Yolo-Auto via .env).
If no LLM is reachable, it degrades gracefully: shows the UNCOMPRESSED
pipeline output plus an actionable hint, then exits 0.

Run from the project root:
    uv run python 06-advanced-retrieval-techniques/code/approach_3_compression_comparison.py
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib
from pathlib import Path

# Make `shared.config` importable no matter which directory we launch from.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm, get_embeddings, get_llm_provider_name

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

TOP_K = 5
QUESTION = ("Which indexing strategy gives sub-millisecond latency "
            "and what are its key parameters?")

COMPRESSION_PROMPT = """You are a context compressor for a RAG system.
Question: {question}

Text:
{text}

Extract ONLY the sentences from the Text that are directly relevant to answering the Question.
Keep the original wording verbatim. Do not add anything. If no sentence is relevant, reply with exactly: NONE"""

OLLAMA_HINT = (
    "[yellow]⚠ Could not reach the LLM, so compression is OFF. Start Ollama locally:\n"
    "    ollama serve\n"
    "    ollama pull llama3.1\n"
    "(or set OPENAI_API_KEY / YOLO_AUTO_API_KEY in .env), then re-run this module.[/yellow]"
)


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — good enough for a delta table."""
    return max(1, len(text) // 4)


def load_enriched_documents() -> list[Document]:
    docs = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def build_vectorstore(docs: list[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d.page_content)):
            meta = dict(d.metadata)
            meta["id"] = f"{d.metadata['source']}::c{i}"
            chunks.append(Document(page_content=piece, metadata=meta))
    console.print("[dim]Embedding chunks…[/dim]")
    return Chroma.from_documents(
        chunks, get_embeddings(), collection_name="m6_compression",
        collection_metadata={"hnsw:space": "cosine"})


def compress_doc(llm, doc: Document) -> str:
    """Ask the LLM to keep only the query-relevant sentences of one doc."""
    out = llm.invoke(COMPRESSION_PROMPT.format(question=QUESTION, text=doc.page_content))
    text = out.content if hasattr(out, "content") else str(out)
    text = text.strip()
    if not text or text.upper().startswith("NONE"):
        return ""
    return text


def print_docs(title: str, docs: list[Document], limit: int = TOP_K) -> None:
    table = Table(title=title, title_style="bold cyan", header_style="bold magenta")
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("source", width=22)
    table.add_column("content", overflow="fold")
    for i, d in enumerate(docs[:limit], 1):
        table.add_row(str(i), d.metadata.get("source", "?"),
                      " ".join(d.page_content.split())[:140])
    console.print(table)


def main() -> None:
    console.rule("[bold]MODULE 6 · CONTEXTUAL COMPRESSION — ON vs OFF")
    console.print(f"[bold]Question:[/bold] {QUESTION}\n")

    try:
        vs = build_vectorstore(load_enriched_documents())
        retrieved = vs.similarity_search(QUESTION, k=TOP_K)
    except Exception as e:  # noqa: BLE001 - friendly top-level handler
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        console.print("[yellow]⚠ Retrieval failed (not an LLM issue). Check `uv sync` "
                      "and that data/samples/ exists.[/yellow]")
        sys.exit(0)

    raw_tokens = sum(approx_tokens(d.page_content) for d in retrieved)
    console.print(f"[green]Retrieved {len(retrieved)} chunks "
                  f"(≈{raw_tokens} tokens total).[/green]\n")

    # ---- Try the LLM; degrade gracefully if unreachable -------------------
    compressed: list[Document] = []
    try:
        llm = get_llm()
        console.print(f"[dim]Compressing each chunk with {get_llm_provider_name()}…[/dim]")
        for d in retrieved:
            kept = compress_doc(llm, d)
            if kept:
                compressed.append(Document(page_content=kept, metadata=dict(d.metadata)))
    except Exception as e:  # noqa: BLE001 - LLM down => show OFF pipeline only
        console.print(Panel(f"[red]{type(e).__name__}: {e}[/red]", title="Compression failed"))
        console.print(OLLAMA_HINT)
        console.print("\n[bold cyan]Pipeline WITHOUT compression (what the generator would see):[/bold cyan]\n")
        print_docs(f"Uncompressed top-{TOP_K} context (≈{raw_tokens} tokens)", retrieved)
        sys.exit(0)

    # ---- Delta table -------------------------------------------------------
    comp_tokens = sum(approx_tokens(d.page_content) for d in compressed)
    saved = raw_tokens - comp_tokens
    pct = (saved / raw_tokens * 100) if raw_tokens else 0.0

    table = Table(title="Token-count delta (raw vs compressed)",
                  title_style="bold cyan", header_style="bold magenta")
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("source", width=22)
    table.add_column("raw tokens", justify="right")
    table.add_column("compressed", justify="right")
    table.add_column("saved %", justify="right")
    comp_by_src = {}
    for d in compressed:
        comp_by_src.setdefault(d.metadata.get("id"), d.page_content)
    for i, d in enumerate(retrieved, 1):
        kept = comp_by_src.get(d.metadata.get("id"))
        c_tok = approx_tokens(kept) if kept else 0
        r_tok = approx_tokens(d.page_content)
        save_pct = f"{(r_tok - c_tok) / r_tok * 100:.0f}%" if r_tok else "-"
        style = "green" if c_tok < r_tok else "red"
        table.add_row(str(i), d.metadata.get("source", "?"), str(r_tok),
                      str(c_tok), f"[{style}]{save_pct}[/{style}]")
    table.add_row("", "[bold]TOTAL[/bold]", f"[bold]{raw_tokens}[/bold]",
                  f"[bold]{comp_tokens}[/bold]",
                  f"[bold green]{pct:.0f}% saved[/bold green]")
    console.print(table)

    # ---- Compressed context ------------------------------------------------
    console.print("\n[bold cyan]Compressed context (what the generator would see):[/bold cyan]\n")
    if compressed:
        for i, d in enumerate(compressed, 1):
            console.print(Panel(
                f"[dim]{d.metadata.get('source', '?')}[/dim]\n\n{d.page_content}",
                border_style="green"))
    else:
        console.print("[yellow]The LLM judged every chunk irrelevant — nothing survived "
                      "compression. The generator would receive an empty context.[/yellow]")

    console.print("\n[dim]Observation: compression trades one extra LLM call per chunk for far "
                  "fewer tokens at generation time. Watch the 'saved %' column — if it's near 0%, "
                  "the chunks were already tight and compression isn't earning its cost.[/dim]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as exc:  # noqa: BLE001 - friendly top-level handler
        console.print(Panel(f"[red]{type(exc).__name__}: {exc}[/red]", title="Startup error"))
        console.print("[dim]Make sure dependencies are installed (`uv sync`) and that the local "
                      "embedding model (all-MiniLM-L6-v2) can be downloaded on first run.[/dim]")
