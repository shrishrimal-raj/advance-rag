"""
HyDE — Approach 2: From Scratch (framework-free)
================================================
Same pattern as main.py (LangChain Chroma), but with ZERO LangChain RAG plumbing:

    LLM writes a HYPOTHETICAL ANSWER DOCUMENT
        -> embed it (MiniLM, local)
        -> retrieve from RAW chromadb using that embedding as the query vector
        -> answer the REAL question using the retrieved context

Why: user questions use everyday words; corpus documents use technical words.
A hypothetical document written in the corpus's own vocabulary lands closer to
the real relevant chunks in vector space than the raw query would.

Run from project root:
    uv run python 07-advanced-rag-patterns/code/hyde/approach_2_from_scratch.py
"""

import sys
import pathlib

# Windows console is cp1252; force UTF-8 output to avoid encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Project root (this file lives in <root>/07-advanced-rag-patterns/code/hyde/)
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

TOP_K = 5          # docs retrieved with the hypothetical-doc embedding
ANSWER_TOP_K = 3   # of those, how many go into the final prompt


# ---------------------------------------------------------------------------
# Stage 0: Corpus loading + raw chromadb index
# ---------------------------------------------------------------------------
def load_corpus() -> list[tuple[str, str]]:
    """Load sample txt/md files and chunk them. Returns (text, source) tuples."""
    data_dir = pathlib.Path(__file__).resolve().parents[3] / "data" / "samples"
    chunks: list[tuple[str, str]] = []
    for file_path in sorted(data_dir.iterdir()):
        if file_path.suffix.lower() not in (".txt", ".md"):
            continue
        content = file_path.read_text(encoding="utf-8")
        for para in content.split("\n\n"):
            para = para.strip()
            if len(para) < 20:
                continue
            for j in range(0, len(para), 500):
                piece = para[j:j + 500].strip()
                if piece:
                    chunks.append((piece, file_path.name))
    return chunks


def build_store(chunks: list[tuple[str, str]]):
    """Build an in-memory chromadb collection with precomputed dense vectors."""
    import chromadb
    from shared.config import get_embeddings

    embeddings = get_embeddings()
    texts = [c[0] for c in chunks]
    vectors = embeddings.embed_documents(texts)  # MiniLM, 384-dim, local & cached

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("hyde_scratch")
    collection.add(
        ids=[f"doc_{i}" for i in range(len(chunks))],
        documents=texts,
        metadatas=[{"source": c[1]} for c in chunks],
        embeddings=vectors,
    )
    return collection, embeddings


def search(collection, embeddings, text: str, k: int = TOP_K):
    """One dense search against raw chromadb. Returns list of doc ids (ranked)."""
    vec = embeddings.embed_query(text)
    res = collection.query(query_embeddings=[vec], n_results=k)
    return res["ids"][0]


# ---------------------------------------------------------------------------
# Stage 1: Hypothetical document (LLM, with graceful fallback)
# ---------------------------------------------------------------------------
def generate_hypothetical_doc(question: str) -> tuple[str, bool]:
    """LLM writes a doc-sounding hypothetical answer. Returns (hyp_doc, llm_ok)."""
    from shared.config import get_llm

    prompt = f"""You are going to write a HYPOTHETICAL ANSWER to the following question.
This hypothetical answer will be used as a SEARCH QUERY in a vector database.

RULES:
- Write it as a technical knowledge-base paragraph about RAG systems, vector
  databases, and AI architecture — not a conversational reply.
- Use domain terminology (vector embeddings, cosine similarity, chunking,
  retrieval, indexing, latency).
- 3-5 sentences. No preamble. Just the content.

Question: {question}

Hypothetical document:"""

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        return response.content.strip(), True
    except Exception as e:
        console.print(f"[red]LLM unavailable ({type(e).__name__}).[/red]")
        console.print("[yellow]Hint: check YOLO_AUTO_API_KEY / OPENAI_API_KEY in .env.[/yellow]")
        # Fallback: use the raw question as the "hypothetical doc" so the
        # retrieval mechanics are still demonstrable (this is just naive RAG).
        return question, False


# ---------------------------------------------------------------------------
# Stage 3: Grounded answer (LLM, with graceful fallback)
# ---------------------------------------------------------------------------
def answer(question: str, docs: list[str]) -> None:
    from shared.config import get_llm

    context = "\n\n".join(f"[Source {i+1}] {d}" for i, d in enumerate(docs))
    prompt = f"""Based ONLY on the retrieved context below, provide a clear, accurate
answer to the question. If the context is insufficient, state what is missing.

Retrieved Context:
{context}

Question: {question}

Answer:"""
    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(prompt)
        console.print(Panel(response.content.strip(), title="[green]Final Grounded Answer[/green]",
                            border_style="green"))
    except Exception as e:
        console.print(f"[red]Could not generate answer ({type(e).__name__}).[/red]")
        console.print("[yellow]Showing the retrieved context instead:[/yellow]")
        for i, d in enumerate(docs, 1):
            console.print(f"[dim][Source {i}] {d[:200]}{'...' if len(d) > 200 else ''}[/dim]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    console.print(Panel("[bold magenta]HyDE — FROM SCRATCH (raw chromadb)[/bold magenta]",
                        subtitle="Hypothetical doc -> embed -> retrieve -> grounded answer"))

    # Deliberately colloquial; the corpus uses technical vocabulary.
    question = "How do I make my search faster and more accurate?"
    console.print(f"\n[bold]User Question:[/bold] {question}")
    console.print("[dim](casual language on purpose — the corpus is technical)[/dim]\n")

    # Stage 0
    chunks = load_corpus()
    console.print(f"Loaded {len(chunks)} chunks from data/samples/")
    try:
        collection, embeddings = build_store(chunks)
    except Exception as e:
        console.print(f"[red]Embeddings/vector store failed: {e}[/red]")
        console.print("[yellow]Hint: local MiniLM should already be cached; check network "
                      "or LOCAL_EMBEDDING_MODEL in .env.[/yellow]")
        return 0

    # Stage 1
    hyp_doc, llm_ok = generate_hypothetical_doc(question)
    console.print(Panel(hyp_doc, title=f"Hypothetical Document ({'LLM' if llm_ok else 'fallback: raw question'})",
                        border_style="yellow"))

    # Stage 2: retrieve with HyDE embedding AND raw query for comparison
    hyde_ids = search(collection, embeddings, hyp_doc)
    direct_ids = search(collection, embeddings, question)

    table = Table(title="HyDE vs Direct Retrieval (raw chromadb)", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("HyDE (hypothetical doc as query)", width=45)
    table.add_column("Direct (raw user query)", width=45)
    for i in range(TOP_K):
        h = collection.get(ids=[hyde_ids[i]])["documents"][0]
        d = collection.get(ids=[direct_ids[i]])["documents"][0]
        table.add_row(str(i + 1), h[:42] + ("..." if len(h) > 42 else ""),
                      d[:42] + ("..." if len(d) > 42 else ""))
    console.print(table)

    # Stage 3
    top_docs = [collection.get(ids=[doc_id])["documents"][0] for doc_id in hyde_ids[:ANSWER_TOP_K]]
    answer(question, top_docs)

    console.print("\n[bold green]Done.[/bold green] Framework-free HyDE complete.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
