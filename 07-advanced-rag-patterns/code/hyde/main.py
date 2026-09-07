"""
HyDE (Hypothetical Document Embeddings)
========================================
Pattern: LLM writes a hypothetical answer -> embed THAT -> retrieve -> generate real answer.

Problem solved: Vocabulary mismatch. When the user's words differ from the document's
words, direct embedding similarity is low. By having the LLM write a "hypothetical
document" that uses the corpus's vocabulary, we bridge the gap.

Why this helps: If the user asks "How do I make search faster?" but the document says
"Query latency is reduced by quantizing HNSW vectors," the raw query embedding is far
from the document embedding. But a hypothetical answer like "To reduce query latency,
one can quantize the HNSW index vectors..." shares vocabulary with the document and
therefore has much higher cosine similarity.

Run from project root:
    uv run python 07-advanced-rag-patterns/code/hyde/main.py
"""

import sys
import pathlib

# Add project root to path so we can import shared.config
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

# Windows console is cp1252; force UTF-8 output to avoid encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.config import get_llm, get_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

console = Console()

TOP_K = 5  # Number of documents to retrieve


# ---------------------------------------------------------------------------
# Stage 0: Load corpus and build vector store
# ---------------------------------------------------------------------------
def load_corpus() -> list[Document]:
    """Load sample text files from data/samples/ and split into chunks."""
    console.print(Panel("[bold cyan]Stage 0: Loading corpus[/bold cyan]"))

    data_dir = pathlib.Path(__file__).resolve().parents[3] / "data" / "samples"
    documents = []

    for file_path in sorted(data_dir.iterdir()):
        if file_path.suffix.lower() in (".txt", ".md"):
            content = file_path.read_text(encoding="utf-8")
            paragraphs = content.split("\n\n")
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) < 20:
                    continue
                # Split long paragraphs into ~500 char chunks
                for j in range(0, len(para), 500):
                    chunk_text = para[j:j+500].strip()
                    if chunk_text:
                        documents.append(Document(
                            page_content=chunk_text,
                            metadata={"source": file_path.name, "chunk_id": i}
                        ))

    console.print(f"  Loaded {len(documents)} chunks.")
    return documents


def build_vector_store(documents: list[Document]) -> Chroma:
    """Build an in-memory Chroma vector store."""
    embeddings = get_embeddings()
    vs = Chroma(embedding_function=embeddings)
    vs.add_documents(documents)
    console.print(f"  Vector store ready ({len(documents)} docs).")
    return vs


# ---------------------------------------------------------------------------
# Stage 1: Generate hypothetical document
# ---------------------------------------------------------------------------
def generate_hypothetical_doc(question: str) -> str:
    """
    Ask the LLM to write a hypothetical answer that sounds like a document
    in the corpus. This bridges the vocabulary gap between user and docs.
    """
    console.print(Panel("[bold cyan]Stage 1: Generating hypothetical document[/bold cyan]"))

    llm = get_llm(temperature=0.3)

    prompt = f"""You are going to write a HYPOTHETICAL ANSWER to the following question.
This hypothetical answer will be used as a SEARCH QUERY in a vector database.

IMPORTANT RULES:
- Write it as if you are a technical document in a knowledge base about RAG systems,
  vector databases, and AI architecture.
- Use domain-specific terminology (vector embeddings, cosine similarity, chunking,
  retrieval, indexing, etc.)
- Make it sound like a paragraph FROM the corpus, not a conversational answer.
- 3-5 sentences. No preamble, no "Here is...". Just the content.

Question: {question}

Hypothetical document:"""

    try:
        response = llm.invoke(prompt)
        hyp_doc = response.content.strip()
        console.print(f"\n  [bold]Hypothetical Document:[/bold]")
        console.print(Panel(hyp_doc, border_style="yellow", width=80))

        # Explain WHY this helps
        console.print("\n  [bold yellow]Why this helps:[/bold yellow]")
        console.print("  The user's question uses everyday language. The documents use")
        console.print("  technical vocabulary. By generating a 'hypothetical document' that")
        console.print("  mirrors the corpus's language, its embedding lands closer to the")
        console.print("  actual relevant documents in vector space than the raw query would.")
        console.print("  This is especially powerful when user vocab != document vocab.\n")
        return hyp_doc

    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        console.print("[yellow]Hint: Make sure Ollama is running (ollama serve) "
                      "and llama3.1 is pulled.[/yellow]")
        # Fallback: use the question itself as the "hypothetical doc"
        return question


# ---------------------------------------------------------------------------
# Stage 2: Retrieve using hypothetical doc as query vector
# ---------------------------------------------------------------------------
def retrieve_with_hyde(vector_store: Chroma, hyp_doc: str, original_query: str):
    """
    Retrieve using the hypothetical document's embedding as the query vector.
    Also retrieve with the original query for comparison.
    """
    console.print(Panel("[bold cyan]Stage 2: Retrieval comparison (HyDE vs Direct)[/bold cyan]"))

    # HyDE retrieval: use hypothetical doc as the search query
    hyde_results = vector_store.similarity_search(hyp_doc, k=TOP_K)

    # Direct retrieval: use original user query (for comparison)
    direct_results = vector_store.similarity_search(original_query, k=TOP_K)

    # Print comparison table
    table = Table(title="HyDE vs Direct Retrieval Comparison", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("HyDE Result (hypothetical doc as query)", width=45)
    table.add_column("Direct Result (raw user query)", width=45)

    for i in range(TOP_K):
        hyde_snippet = hyde_results[i].page_content[:42] + "..." if len(hyde_results[i].page_content) > 45 else hyde_results[i].page_content
        direct_snippet = direct_results[i].page_content[:42] + "..." if len(direct_results[i].page_content) > 45 else direct_results[i].page_content
        table.add_row(str(i+1), hyde_snippet, direct_snippet)

    console.print(table)
    console.print("\n  [dim]Notice how HyDE results tend to use more technical language[/dim]")
    console.print("  [dim]matching the corpus, while direct results may be less precise.[/dim]\n")

    return hyde_results


# ---------------------------------------------------------------------------
# Stage 3: Generate grounded answer
# ---------------------------------------------------------------------------
def generate_answer(question: str, retrieved_docs: list[Document]) -> str:
    """Generate a real, grounded answer using the retrieved chunks."""
    console.print(Panel("[bold cyan]Stage 3: Generating grounded answer[/bold cyan]"))

    llm = get_llm(temperature=0.0)

    context = "\n\n".join([f"[Source {i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)])

    prompt = f"""Based ONLY on the retrieved context below, provide a clear, accurate answer
to the question. Cite which source supports each claim. If the context is insufficient,
state what is missing.

Retrieved Context:
{context}

Question: {question}

Answer:"""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        console.print(Panel(answer, title="[green]Final Grounded Answer[/green]", border_style="green"))
        return answer
    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        console.print("[yellow]Hint: Make sure Ollama is running (ollama serve) "
                      "and llama3.1 is pulled.[/yellow]")
        return "[Could not generate answer - LLM unavailable]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.print(Panel("[bold magenta]HyDE - HYPOTHETICAL DOCUMENT EMBEDDINGS[/bold magenta]",
                        subtitle="Bridge vocabulary gaps with LLM-generated search docs"))

    # User question deliberately uses colloquial language different from corpus
    question = "How do I make my search faster and more accurate?"

    console.print(f"\n[bold]User Question:[/bold] {question}")
    console.print("[dim](Note: this uses casual language; the corpus uses technical terms)[/dim]\n")

    # Stage 0: Load corpus
    documents = load_corpus()
    vector_store = build_vector_store(documents)

    # Stage 1: Generate hypothetical document
    hyp_doc = generate_hypothetical_doc(question)

    # Stage 2: Retrieve with HyDE (and compare with direct)
    hyde_results = retrieve_with_hyde(vector_store, hyp_doc, question)

    # Stage 3: Generate final answer
    generate_answer(question, hyde_results[:3])  # Use top 3 for context

    console.print("\n[bold green]Done.[/bold green] HyDE pipeline complete.\n")


if __name__ == "__main__":
    main()
