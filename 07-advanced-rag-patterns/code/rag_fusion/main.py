"""
RAG Fusion + Reciprocal Rank Fusion (RRF)
==========================================
Pattern: Generate multiple query variants -> retrieve independently -> fuse with RRF.

Problem solved: A single query may miss relevant documents due to vocabulary/phrasing
differences. By generating 3 variants and fusing their results, we get a more robust
retrieval that's less sensitive to any single query's wording.

Run from project root:
    uv run python 07-advanced-rag-patterns/code/rag_fusion/main.py
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
from rich.text import Text

from shared.config import get_llm, get_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RRF_K = 60          # RRF constant (higher = less weight on top ranks)
NUM_VARIANTS = 3    # Number of query variants to generate
TOP_K_PER_QUERY = 5 # Retrieve this many docs per variant
FINAL_TOP_K = 3     # Use top-N fused results for answer generation


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
            # Simple chunking: split by double newlines, then cap at ~500 chars
            paragraphs = content.split("\n\n")
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) < 20:
                    continue
                # If paragraph is very long, split further
                chunks = []
                for j in range(0, len(para), 500):
                    chunk_text = para[j:j+500].strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                for c in chunks:
                    documents.append(Document(
                        page_content=c,
                        metadata={"source": file_path.name, "chunk_id": i}
                    ))

    console.print(f"  Loaded {len(documents)} chunks from {data_dir.name}/")
    return documents


def build_vector_store(documents: list[Document]) -> Chroma:
    """Build an in-memory Chroma vector store."""
    embeddings = get_embeddings()
    vs = Chroma(embedding_function=embeddings)
    vs.add_documents(documents)
    console.print(f"  Vector store built with {len(documents)} documents.")
    return vs


# ---------------------------------------------------------------------------
# Stage 1: Generate query variants using LLM
# ---------------------------------------------------------------------------
def generate_query_variants(question: str) -> list[str]:
    """Ask the LLM to generate NUM_VARIANTS rephrasings of the user question."""
    console.print(Panel("[bold cyan]Stage 1: Generating query variants[/bold cyan]"))

    llm = get_llm(temperature=0.7)  # Slightly higher temp for diversity

    prompt = f"""Given the question: "{question}"

Generate {NUM_VARIANTS} different ways to rephrase this question for document retrieval.
Each variant should use different words, angles, or levels of specificity.
Output ONLY the {NUM_VARIANTS} queries, one per line, no numbering, no extra text."""

    try:
        response = llm.invoke(prompt)
        lines = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
        # Take first NUM_VARIANTS lines
        variants = lines[:NUM_VARIANTS]
        # Ensure we have at least the original question
        if not variants:
            variants = [question]
        while len(variants) < NUM_VARIANTS:
            variants.append(question)  # Pad with original if LLM gave fewer

        console.print(f"  Generated {len(variants)} query variants:")
        for i, v in enumerate(variants, 1):
            console.print(f"    [{i}] {v}")
        return variants

    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        console.print("[yellow]Hint: Make sure Ollama is running (ollama serve) "
                      "and llama3.1 is pulled.[/yellow]")
        # Fallback: just use the original question repeated
        return [question] * NUM_VARIANTS


# ---------------------------------------------------------------------------
# Stage 2: Retrieve top-k for each variant
# ---------------------------------------------------------------------------
def retrieve_per_variant(vector_store: Chroma, variants: list[str]) -> list[list[Document]]:
    """Retrieve TOP_K_PER_QUERY documents for each query variant independently."""
    console.print(Panel("[bold cyan]Stage 2: Retrieving per variant[/bold cyan]"))

    all_results = []
    for i, variant in enumerate(variants, 1):
        results = vector_store.similarity_search(variant, k=TOP_K_PER_QUERY)
        all_results.append(results)
        console.print(f"  Variant [{i}] retrieved {len(results)} docs. "
                      f"Top: '{results[0].page_content[:60]}...'")

    return all_results


# ---------------------------------------------------------------------------
# Stage 3: Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(all_results: list[list[Document]], k: int = RRF_K) -> list[tuple[Document, float, list[int]]]:
    """
    Implement Reciprocal Rank Fusion inline.

    For each document, compute:
        RRF_score(doc) = sum over all lists of: 1 / (k + rank_in_that_list)

    where rank is 1-indexed. Documents not in a list contribute 0.

    Returns: list of (document, rrf_score, [ranks_in_each_list]) sorted by score desc.
    """
    console.print(Panel("[bold cyan]Stage 3: Reciprocal Rank Fusion (k={})[/bold cyan]".format(k)))

    # Map: doc_content -> {score, ranks, doc_object}
    doc_scores: dict[str, dict] = {}

    for list_idx, results in enumerate(all_results):
        for rank, doc in enumerate(results, start=1):  # 1-indexed rank
            key = doc.page_content  # Use content as unique identifier
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0, "ranks": [None] * len(all_results)}
            doc_scores[key]["score"] += 1.0 / (k + rank)
            doc_scores[key]["ranks"][list_idx] = rank

    # Sort by RRF score descending
    fused = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)

    # Format output
    output = []
    for item in fused:
        output.append((item["doc"], item["score"], item["ranks"]))

    console.print(f"  Fused {len(output)} unique documents from {len(all_results)} lists.")
    return output


# ---------------------------------------------------------------------------
# Stage 4: Print fused ranking table
# ---------------------------------------------------------------------------
def print_fused_ranking(fused: list[tuple[Document, float, list[int]]]):
    """Print a nice table showing the fused ranking with per-query rank columns."""
    table = Table(title="Fused Ranking (RRF)", show_lines=True)
    table.add_column("Rank", style="bold", width=5)
    table.add_column("Document (truncated)", width=50)
    table.add_column("RRF Score", justify="right")
    for i in range(len(fused[0][2]) if fused else 0):
        table.add_column(f"Q{i+1} Rank", justify="center", width=9)

    for rank, (doc, score, ranks) in enumerate(fused[:8], 1):  # Show top 8
        snippet = doc.page_content[:47] + "..." if len(doc.page_content) > 50 else doc.page_content
        row_ranks = [str(r) if r is not None else "-" for r in ranks]
        table.add_row(str(rank), snippet, f"{score:.4f}", *row_ranks)

    console.print(table)


# ---------------------------------------------------------------------------
# Stage 5: Generate final answer from fused context
# ---------------------------------------------------------------------------
def generate_answer(question: str, fused_docs: list[Document]) -> str:
    """Use the top fused documents as context to generate an answer."""
    console.print(Panel("[bold cyan]Stage 5: Generating answer from fused context[/bold cyan]"))

    llm = get_llm(temperature=0.0)

    context = "\n\n".join([f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(fused_docs)])

    prompt = f"""Using ONLY the following context, answer the question.
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        console.print(Panel(answer, title="[green]Final Answer[/green]", border_style="green"))
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
    console.print(Panel("[bold magenta]RAG FUSION + RECIPROCAL RANK FUSION[/bold magenta]",
                        subtitle="Multi-query fan-out with RRF merge"))

    # The user's question
    question = "How does vector-based retrieval work in a RAG system?"

    console.print(f"\n[bold]User Question:[/bold] {question}\n")

    # Stage 0: Load corpus
    documents = load_corpus()
    vector_store = build_vector_store(documents)

    # Stage 1: Generate query variants
    variants = generate_query_variants(question)

    # Stage 2: Retrieve per variant
    all_results = retrieve_per_variant(vector_store, variants)

    # Stage 3: RRF fusion
    fused = reciprocal_rank_fusion(all_results)

    # Stage 4: Print ranking
    print_fused_ranking(fused)

    # Stage 5: Generate answer
    top_docs = [doc for doc, _, _ in fused[:FINAL_TOP_K]]
    generate_answer(question, top_docs)

    console.print("\n[bold green]Done.[/bold green] RAG Fusion complete.\n")


if __name__ == "__main__":
    main()
