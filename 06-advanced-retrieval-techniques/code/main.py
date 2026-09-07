"""
Module 6 — Advanced Retrieval Techniques
=========================================
Builds a metadata-enriched corpus from data/samples/ and demonstrates FOUR
advanced retrievers:

  1. Contextual Compression Retriever  (retrieve more, LLM extracts the gold)
  2. Parent-Document Retriever         (small-to-big: precise child, full parent)
  3. Self-Query Retriever              (NL question -> query + metadata filters)
  4. Multi-Query Retriever             (LLM rewrites the question N ways)

Retrievers 1, 3 and 4 use an LLM (local Ollama by default). If Ollama is not
running, each one degrades gracefully with a friendly message instead of crashing.
Retriever 2 (Parent-Document) is pure vector search and always works.

Run from the project root:
    uv run python 06-advanced-retrieval-techniques/code/main.py
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib
from pathlib import Path

# Make `shared.config` importable no matter which directory we launch from.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm, get_embeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# Version-tolerant imports: LangChain <1.0 kept these in `langchain.retrievers`;
# LangChain >=1.0 moved them to `langchain_classic.retrievers`.
try:
    from langchain_classic.retrievers import (
        ContextualCompressionRetriever,
        ParentDocumentRetriever,
        SelfQueryRetriever,
        MultiQueryRetriever,
    )
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
except ImportError:  # pragma: no cover - older LangChain installs
    from langchain.retrievers import (
        ContextualCompressionRetriever,
        ParentDocumentRetriever,
        SelfQueryRetriever,
        MultiQueryRetriever,
    )
    from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.stores import InMemoryStore  # moved out of langchain.storage in LC 1.x
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

# High-level topic per source file -> gives Self-Query a meaningful field to filter on.
TOPICS = {
    "rag_overview.txt": "RAG fundamentals",
    "vector_db_notes.md": "Vector databases & indexing",
    "company_profile.json": "Company profile & robotics products",
    "products.csv": "Product catalog & pricing",
}

OLLAMA_HINT = (
    "[yellow]⚠ Could not reach the LLM. Start Ollama locally:\n"
    "    ollama serve\n"
    "    ollama pull llama3.1\n"
    "then re-run this module.[/yellow]"
)


def fail(err: Exception, needs_llm: bool = True) -> None:
    """Print a friendly, non-crashing error for a failed retriever step."""
    if needs_llm:
        console.print(OLLAMA_HINT)
    else:
        console.print("[yellow]⚠ This step failed (not an LLM issue).[/yellow]")
    console.print(f"[red]({type(err).__name__}: {err})[/red]\n")


# ---------------------------------------------------------------------------
# 1. BUILD THE METADATA-ENRICHED CORPUS
# ---------------------------------------------------------------------------
def load_enriched_documents() -> list[Document]:
    """Read every sample file, tagging source / format / topic metadata."""
    docs = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source": path.name,
                "format": path.suffix.lstrip(".").lower(),
                "topic": TOPICS.get(path.name, "general"),
            },
        ))
    return docs


def build_main_vectorstore(docs: list[Document]) -> tuple[Chroma, list[Document]]:
    """Chunk the enriched docs and embed into an in-memory Chroma collection."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d.page_content)):
            meta = dict(d.metadata)
            meta["id"] = f"{d.metadata['source']}::c{i}"
            chunks.append(Document(page_content=piece, metadata=meta))
    console.print("[dim]Embedding enriched chunks…[/dim]")
    vs = Chroma.from_documents(
        chunks, get_embeddings(), collection_name="m6_main",
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vs, chunks


# ---------------------------------------------------------------------------
# 2. SELF-QUERY SCHEMA (the metadata the LLM may filter on)
# ---------------------------------------------------------------------------
class DocSchema(BaseModel):
    """Metadata schema exposed to the Self-Query LLM."""
    source: str = Field(description="Source filename, e.g. 'products.csv'")
    format: str = Field(description="File format: txt, md, csv or json")
    topic: str = Field(description="High-level topic of the document")


# ---------------------------------------------------------------------------
# 3. PRETTY PRINTING
# ---------------------------------------------------------------------------
def show_docs(title: str, docs: list[Document], limit: int = 3) -> None:
    table = Table(title=title, title_style="bold cyan", header_style="bold magenta")
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("source", width=22)
    table.add_column("topic", width=26)
    table.add_column("content", overflow="fold")
    for i, d in enumerate(docs[:limit], 1):
        table.add_row(
            str(i),
            d.metadata.get("source", "?"),
            d.metadata.get("topic", "?"),
            " ".join(d.page_content.split())[:140],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# 4. THE FOUR ADVANCED RETRIEVERS
# ---------------------------------------------------------------------------
def demo_compression(vs: Chroma) -> None:
    console.print(Panel(
        "[bold]1 · CONTEXTUAL COMPRESSION RETRIEVER[/bold]\n"
        "Retrieve k=4, then let the LLM keep ONLY the sentences that answer the question.",
        border_style="green"))
    question = ("Which indexing strategy gives sub-millisecond latency "
                "and what are its key parameters?")
    try:
        # from_llm() factory works on both old (langchain) and new
        # (langchain_classic) APIs; the plain constructor changed shape.
        compressor = LLMChainExtractor.from_llm(get_llm())
        base = vs.as_retriever(search_kwargs={"k": 4})
        comp = ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)
        docs = comp.invoke(question)
        console.print(f"[bold]Question:[/bold] {question}\n")
        if docs:
            show_docs("Compressed context (LLM-extracted sentences)", docs, limit=4)
        else:
            console.print("[yellow]LLM extracted no relevant sentences.[/yellow]")
        console.print("\n[dim]Observation: the returned text is far shorter than the raw chunks — "
                      "fewer tokens sent to the generator, at the cost of one extra LLM call.[/dim]\n")
    except Exception as e:  # noqa: BLE001
        fail(e, needs_llm=True)


def demo_parent_document(docs: list[Document]) -> None:
    console.print(Panel(
        "[bold]2 · PARENT-DOCUMENT RETRIEVER (small-to-big)[/bold]\n"
        "Index tiny 200-char children for precise matching, return the full 1000-char parent.",
        border_style="green"))
    question = "Explain how HNSW builds its multi-layer graph."
    try:
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        parent_store = InMemoryStore()
        child_vs = Chroma(embedding_function=get_embeddings(), collection_name="m6_children",
                          collection_metadata={"hnsw:space": "cosine"})
        # LC>=1.0 renamed parent_doc_store -> byte_store; pick by inspecting
        # the model so the same code runs on both LangChain generations.
        store_kwarg = (
            {"byte_store": parent_store}
            if "byte_store" in ParentDocumentRetriever.model_fields
            else {"parent_doc_store": parent_store}
        )
        retriever = ParentDocumentRetriever(
            vectorstore=child_vs,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            **store_kwarg,
        )
        retriever.add_documents(docs)
        parents = retriever.invoke(question)
        console.print(f"[bold]Question:[/bold] {question}\n")
        if parents:
            show_docs("Returned PARENT documents (full context)", parents, limit=2)
            console.print(f"\n[dim]Observation: children (≈200 chars) were matched, but we received "
                          f"the larger parent (≈1000 chars) so the generator sees surrounding context.[/dim]\n")
        else:
            console.print("[yellow]No parent documents returned.[/yellow]\n")
    except Exception as e:  # noqa: BLE001
        fail(e, needs_llm=False)


def demo_self_query(vs: Chroma) -> None:
    console.print(Panel(
        "[bold]3 · SELF-QUERY RETRIEVER[/bold]\n"
        "LLM turns a natural-language question into a search query + metadata filters.",
        border_style="green"))
    questions = [
        "Show me electronics products from the CSV catalog.",
        "What does the company profile say about support coverage?",
    ]
    try:
        # LC>=1.0 signature needs document_contents + metadata_field_info;
        # older versions used document_prompt=<pydantic schema>.
        sample_text = "\n".join((vs.get() or {}).get("documents", [])[:20])
        # Upstream quirk: literal braces in document_contents (our JSON sample
        # chunks contain "{...}") make FewShotPromptTemplate's f-string
        # validation raise "Nested replacement fields are not allowed".
        # Neutralize braces before handing the text to the prompt builder.
        sample_text = sample_text.replace("{", " [").replace("}", " ]")
        field_info = [
            {"name": "format", "type": "str",
             "description": "source file format: txt, md, csv or json"},
            {"name": "topic", "type": "str",
             "description": "high-level topic of the chunk"},
        ]
        try:
            sq = SelfQueryRetriever.from_llm(
                llm=get_llm(), vectorstore=vs,
                document_contents=sample_text, metadata_field_info=field_info,
            )
        except TypeError:  # pragma: no cover - older LangChain
            sq = SelfQueryRetriever.from_llm(
                llm=get_llm(), vectorstore=vs, document_prompt=DocSchema)
        for q in questions:
            console.print(f"\n[bold]Question:[/bold] {q}")
            docs = sq.invoke(q)
            show_docs("Filtered results", docs, limit=3)
        console.print("\n[dim]Observation: the LLM added structured filters (e.g. format='csv') so the "
                      "vector search only scans matching documents.[/dim]\n")
    except Exception as e:  # noqa: BLE001
        fail(e, needs_llm=True)


def demo_multi_query(vs: Chroma) -> None:
    console.print(Panel(
        "[bold]4 · MULTI-QUERY RETRIEVER (num_queries=3)[/bold]\n"
        "LLM rewrites the question 3 ways, retrieves per query, dedupes & merges.",
        border_style="green"))
    question = "Tell me about Acme Robotics and its products."
    try:
        base = vs.as_retriever(search_kwargs={"k": 2})
        mq = MultiQueryRetriever.from_llm(llm=get_llm(), retriever=base)
        docs = mq.invoke(question)
        console.print(f"[bold]Question:[/bold] {question}\n")
        show_docs("Merged results from 3 rewritten queries (deduped)", docs, limit=5)
        console.print("\n[dim]Observation: ambiguous/broad questions benefit — 3 phrasings cast a wider "
                      "net, then duplicates are removed.[/dim]\n")
    except Exception as e:  # noqa: BLE001
        fail(e, needs_llm=True)


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    console.rule("[bold]MODULE 6 · ADVANCED RETRIEVAL TECHNIQUES")
    docs = load_enriched_documents()
    console.print(f"[green]Loaded {len(docs)} enriched source documents.[/green]")
    vs, chunks = build_main_vectorstore(docs)
    console.print(f"[green]Indexed {len(chunks)} chunks (with source/format/topic metadata).[/green]\n")

    demo_compression(vs)
    demo_parent_document(docs)
    demo_self_query(vs)
    demo_multi_query(vs)

    console.rule("[bold]DONE")
    console.print("[dim]Tip: retrievers 1, 3 and 4 need a running LLM (Ollama). "
                  "Retriever 2 (parent-document) is pure vector search and always works.[/dim]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as exc:  # noqa: BLE001 - friendly top-level handler
        console.print(Panel(f"[red]{type(exc).__name__}: {exc}[/red]", title="Startup error"))
        console.print("[dim]Make sure dependencies are installed (`uv sync`) and that the local "
                      "embedding model (all-MiniLM-L6-v2) can be downloaded on first run.[/dim]")
