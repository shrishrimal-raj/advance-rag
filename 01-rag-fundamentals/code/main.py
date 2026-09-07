"""Module 1 — Naive RAG Pipeline (end-to-end).

Flow: Load docs -> Chunk -> Embed -> ChromaDB -> Retrieve top-k -> Generate grounded answer.

Run from project root:
    uv run python 01-rag-fundamentals/code/main.py
"""
import sys
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel

console = Console()

EXAMPLE_QUESTIONS = [
    "What is Retrieval-Augmented Generation and what problems does it solve?",
    "Which indexing strategies are mentioned for vector databases?",
    "Why is hybrid search useful compared to pure vector search?",
]


def load_documents():
    """Step 1: Load sample documents into LangChain Document objects."""
    from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader

    samples = pathlib.Path(__file__).resolve().parents[2] / "data" / "samples"
    docs = []
    txt = samples / "rag_overview.txt"
    md = samples / "vector_db_notes.md"
    if txt.exists():
        docs += TextLoader(txt, encoding="utf-8").load()
    if md.exists():
        docs += UnstructuredMarkdownLoader(md).load()
    console.print(f"[bold green]✓ Loaded {len(docs)} documents[/bold green]")
    return docs


def chunk_documents(docs):
    """Step 2: Split into overlapping chunks (~300 chars, 50 overlap)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    console.print(f"[bold green]✓ Created {len(chunks)} chunks[/bold green]")
    return chunks


def build_vector_store(chunks):
    """Step 3: Embed chunks and store in an in-memory Chroma collection."""
    from langchain_chroma import Chroma
    from shared.config import get_embeddings

    vs = Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name="module1_naive_rag",
    )
    console.print(f"[bold green]✓ Vector store ready ({len(chunks)} vectors)[/bold green]")
    return vs


def ask(vs, question: str) -> None:
    """Steps 4+5: Retrieve top-3, then generate a grounded, cited answer."""
    from shared.config import get_llm

    console.rule(f"[bold cyan]Q: {question}[/bold cyan]")

    # --- RETRIEVE ---
    # similarity_search_with_score returns [(Document, distance), ...]
    # (Chroma scores are distances: LOWER = more similar)
    try:
        scored = vs.similarity_search_with_score(question, k=3)
    except Exception as e:
        console.print(f"[red]Retrieval failed: {e}[/red]")
        return
    docs = [d for d, _ in scored]
    scores = {id(d): s for d, s in scored}

    if not docs:
        console.print("[yellow]No relevant chunks found.[/yellow]\n")
        return

    console.print("[dim]--- Retrieved chunks ---[/dim]")
    context_blocks = []
    for i, d in enumerate(docs, 1):
        score = scores.get(id(d))
        src = d.metadata.get("source", "?")
        console.print(f"[dim][{i}] (dist={score:.3f}, src={src}) {d.page_content[:160]}...[/dim]")
        context_blocks.append(f"[{i}] (source: {src})\n{d.page_content}")

    # --- GENERATE ---
    prompt = (
        "You are a precise assistant. Use ONLY the numbered context below to answer.\n"
        "Cite sources inline like [1], [2]. If the context is insufficient, say so.\n\n"
        "CONTEXT:\n" + "\n\n".join(context_blocks) + f"\n\nQUESTION: {question}\n\nANSWER:"
    )
    try:
        llm = get_llm()
        answer = llm.invoke(prompt).content
    except Exception as e:
        console.print(
            f"[yellow]LLM unavailable ({type(e).__name__}). "
            "Start Ollama: 'ollama serve' + 'ollama pull llama3.1', or set OPENAI_API_KEY in .env.[/yellow]"
        )
        return

    console.print(Panel(answer, title="Answer", border_style="green"))
    console.print()


def main():
    console.print(Panel("[bold]Module 1 — Naive RAG Pipeline[/bold]", border_style="blue"))
    docs = load_documents()
    chunks = chunk_documents(docs)
    vs = build_vector_store(chunks)

    for q in EXAMPLE_QUESTIONS:
        ask(vs, q)

    console.rule("[bold red]Why this pipeline is NAIVE[/bold red]")
    console.print(
        """
• Single dense retriever — misses exact keyword matches (fix: hybrid BM25+dense, Module 5)
• Fixed top-k, no relevance threshold — may ground answers in weak chunks (Module 5)
• No reranking — initial ranking order is often wrong (Module 7/10)
• No evaluation — you cannot measure whether changes help (Module 9)
• No guardrails/caching/monitoring — not production-ready (Module 11)
"""
    )


if __name__ == "__main__":
    main()
