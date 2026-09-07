"""Module 1 — Approach 2: Step-Back Prompting RAG.

Idea (Zheng et al., 2023): concrete questions ("What is the HNSW ef_search knob?")
often retrieve poorly because the exact phrasing is rare in the corpus. Step-back
RAG first asks the LLM to *abstract* the question into a broader topic
("How do vector database indexing parameters work?"), retrieves on THAT, then
answers the ORIGINAL question with the retrieved context. Broader queries match
more chunks → better grounding for specific questions.

Flow per question:
    Naive:     Q ──embed──> retrieve top-k ──> answer(Q, ctx)
    Step-back: Q ──LLM──> Q' (abstracted) ──embed──> retrieve top-k ──> answer(Q, ctx)

Run from project root:
    uv run python 01-rag-fundamentals/code/approach_2_stepback_rag.py
"""
import sys
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

EXAMPLE_QUESTIONS = [
    "Which indexing strategies are mentioned for vector databases?",
    "Why is hybrid search useful compared to pure vector search?",
]


def load_documents():
    """Load sample documents into LangChain Document objects."""
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
    """Split into overlapping chunks (~300 chars, 50 overlap)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    console.print(f"[bold green]✓ Created {len(chunks)} chunks[/bold green]")
    return chunks


def build_vector_store(chunks):
    """Embed chunks into an in-memory Chroma collection."""
    from langchain_chroma import Chroma
    from shared.config import get_embeddings

    vs = Chroma.from_documents(chunks, get_embeddings(), collection_name="module1_stepback_rag")
    console.print(f"[bold green]✓ Vector store ready ({len(chunks)} vectors)[/bold green]")
    return vs


def get_llm_or_exit():
    """Probe the LLM once; on failure print an actionable hint and exit 0."""
    from shared.config import get_llm, get_llm_provider_name

    try:
        llm = get_llm()
        llm.invoke("Reply with the single word: ok")
        console.print(f"[bold green]✓ LLM reachable:[/bold green] {get_llm_provider_name()}")
        return llm
    except Exception as e:
        console.print(
            Panel(
                f"[yellow]LLM unavailable ({type(e).__name__}: {e}).\n\n"
                "This script needs an LLM for BOTH the step-back abstraction and the answer.\n"
                "Fix one of these and re-run:\n"
                "  • Local:  ollama serve && ollama pull llama3.1\n"
                "  • Cloud:  set OPENAI_API_KEY or YOLO_AUTO_API_KEY in .env[/yellow]",
                title="⚠ Graceful degradation",
                border_style="yellow",
            )
        )
        sys.exit(0)


def step_back_question(llm, question: str) -> str:
    """Ask the LLM to abstract a concrete question into a broader topic question."""
    prompt = (
        "You are a question-rewriter. Given a specific question, produce ONE more "
        "abstract 'step-back' question about the broader topic it belongs to. "
        "The step-back question must be answerable from general documentation. "
        "Output ONLY the step-back question, nothing else.\n\n"
        f"Specific question: {question}\n\nStep-back question:"
    )
    try:
        out = llm.invoke(prompt).content.strip()
        # Strip common quote/label prefixes the model may add
        for prefix in ('Step-back question:', 'Step-back:', '"', "'"):
            if out.startswith(prefix):
                out = out[len(prefix):].strip()
        return out.strip('"\'') or question
    except Exception as e:
        console.print(f"[yellow]Step-back rewrite failed ({type(e).__name__}), using original.[/yellow]")
        return question


def retrieve(vs, text: str, k: int = 3):
    """Return [(Document, distance), ...] — lower distance = more similar."""
    try:
        return vs.similarity_search_with_score(text, k=k)
    except Exception as e:
        console.print(f"[red]Retrieval failed: {e}[/red]")
        return []


def generate(llm, question: str, scored_docs) -> str:
    """Grounded answer with inline citations [1], [2], ..."""
    if not scored_docs:
        return "(no relevant chunks retrieved)"
    blocks = []
    for i, (d, dist) in enumerate(scored_docs, 1):
        src = d.metadata.get("source", "?")
        blocks.append(f"[{i}] (dist={dist:.3f}, source: {src})\n{d.page_content}")
    prompt = (
        "You are a precise assistant. Use ONLY the numbered context below to answer.\n"
        "Cite sources inline like [1], [2]. If the context is insufficient, say so.\n\n"
        "CONTEXT:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}\n\nANSWER:"
    )
    return llm.invoke(prompt).content.strip()


def compare(vs, llm, question: str) -> None:
    """Run naive RAG and step-back RAG on the same question, show side by side."""
    console.rule(f"[bold cyan]Q: {question}[/bold cyan]")

    # --- Naive RAG: retrieve on the original question ---
    naive_docs = retrieve(vs, question)
    console.print("[dim]Naive retrieval query: [/dim]" + question[:120])
    naive_answer = generate(llm, question, naive_docs)

    # --- Step-back RAG: abstract first, retrieve on the abstraction ---
    sb_question = step_back_question(llm, question)
    console.print("[dim]Step-back retrieval query: [/dim]" + sb_question[:120])
    sb_docs = retrieve(vs, sb_question)
    sb_answer = generate(llm, question, sb_docs)

    def fmt_docs(scored_docs) -> str:
        if not scored_docs:
            return "[dim](none)[/dim]"
        lines = []
        for i, (d, dist) in enumerate(scored_docs, 1):
            src = pathlib.Path(d.metadata.get("source", "?")).name
            lines.append(f"[{i}] dist={dist:.3f} · {src}\n[dim]{d.page_content[:110].strip()}…[/dim]")
        return "\n".join(lines)

    table = Table(title=f"Naive RAG vs Step-Back RAG — “{question[:40]}…”", expand=True)
    table.add_column("Naive RAG", ratio=1, style="blue")
    table.add_column("Step-Back RAG", ratio=1, style="magenta")
    table.add_row(
        f"[bold]Retrieval query:[/bold]\n{question}",
        f"[bold]Retrieval query:[/bold]\n{sb_question}",
    )
    table.add_row("[bold]Top-3 retrieved:[/bold]\n" + fmt_docs(naive_docs),
                  "[bold]Top-3 retrieved:[/bold]\n" + fmt_docs(sb_docs))
    table.add_row("[bold]Answer:[/bold]\n" + naive_answer,
                  "[bold]Answer:[/bold]\n" + sb_answer)
    console.print(table)
    console.print()


def main():
    console.print(Panel("[bold]Module 1 — Approach 2: Step-Back Prompting RAG[/bold]", border_style="blue"))
    llm = get_llm_or_exit()
    docs = load_documents()
    chunks = chunk_documents(docs)
    vs = build_vector_store(chunks)

    for q in EXAMPLE_QUESTIONS:
        compare(vs, llm, q)

    console.rule("[bold red]When does step-back help (and hurt)?[/bold red]")
    console.print(
        """
• Helps: specific questions whose exact wording is rare in the corpus — the abstract
  query matches more chunks and pulls in the surrounding explanation.
• Hurts: already-broad questions (abstraction adds nothing) and very long-tail facts
  where the broader context dilutes the one precise chunk you needed.
• Cost: +1 LLM call per question (latency & tokens). Production systems often make
  step-back conditional (e.g., only when naive retrieval scores are low).
"""
    )


if __name__ == "__main__":
    main()
