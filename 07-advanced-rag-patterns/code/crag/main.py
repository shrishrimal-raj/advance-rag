"""
CRAG (Corrective RAG)
======================
Pattern: Retrieve -> Validate each doc (Correct/Incorrect/Ambiguous) -> 
         if good: answer from docs; if bad: web fallback; if web fails: degrade.

Problem solved: Retrieved chunks are not always relevant. Basic RAG blindly stuffs
whatever comes back into the prompt. CRAG adds a quality gate that can reject bad
retrieval results and fall back to the open web or admit ignorance.

Run from project root:
    uv run python 07-advanced-rag-patterns/code/crag/main.py
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

TOP_K = 3  # Number of documents to retrieve for validation
CORRECT_THRESHOLD = 2  # Need at least this many "Correct" docs to skip web fallback


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
# Stage 1: Retrieve top-K documents
# ---------------------------------------------------------------------------
def retrieve_documents(vector_store: Chroma, question: str) -> list[Document]:
    """Retrieve top-K documents for the question."""
    console.print(Panel("[bold cyan]Stage 1: Retrieving top-{} documents[/bold cyan]".format(TOP_K)))

    results = vector_store.similarity_search(question, k=TOP_K)
    for i, doc in enumerate(results, 1):
        snippet = doc.page_content[:70] + "..." if len(doc.page_content) > 70 else doc.page_content
        console.print(f"  [{i}] ({doc.metadata.get('source', '?')}) {snippet}")

    return results


# ---------------------------------------------------------------------------
# Stage 2: LLM Knowledge Validator
# ---------------------------------------------------------------------------
def validate_document(question: str, doc: Document) -> str:
    """
    Ask the LLM to classify a document as Correct, Incorrect, or Ambiguous
    with respect to the user's question.

    Returns: "Correct", "Incorrect", or "Ambiguous"
    """
    llm = get_llm(temperature=0.0)

    prompt = f"""You are a strict knowledge validator. Classify the following document
with respect to the user's question.

Question: {question}

Document:
{doc.page_content}

Classification rules:
- CORRECT: The document directly and sufficiently answers the question.
- INCORRECT: The document is unrelated, contradictory, or does not address the question.
- AMBIGUOUS: The document is partially relevant but incomplete or needs more context.

Respond with EXACTLY one word: Correct, Incorrect, or Ambiguous.
No explanation, no punctuation, just the word."""

    try:
        response = llm.invoke(prompt)
        classification = response.content.strip().lower()
        # Normalize the response
        if "correct" in classification and "incorrect" not in classification:
            return "Correct"
        elif "incorrect" in classification:
            return "Incorrect"
        elif "ambiguous" in classification:
            return "Ambiguous"
        else:
            # Default to Ambiguous if we can't parse
            return "Ambiguous"
    except Exception as e:
        console.print(f"  [red]Validation LLM call failed: {e}[/red]")
        return "Ambiguous"


def validate_all_documents(question: str, docs: list[Document]) -> list[tuple[Document, str]]:
    """Validate all retrieved documents and print decisions."""
    console.print(Panel("[bold cyan]Stage 2: Knowledge Validation[/bold cyan]"))

    table = Table(title="Validation Decisions", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Document (truncated)", width=50)
    table.add_column("Verdict", width=12)
    table.add_column("Style", width=5)

    validated = []
    for i, doc in enumerate(docs, 1):
        verdict = validate_document(question, doc)
        validated.append((doc, verdict))

        # Color code the verdict
        style = {"Correct": "green", "Incorrect": "red", "Ambiguous": "yellow"}[verdict]
        snippet = doc.page_content[:47] + "..." if len(doc.page_content) > 50 else doc.page_content
        table.add_row(str(i), snippet, f"[{style}]{verdict}[/{style}]", "")

    console.print(table)

    # Count verdicts
    correct_count = sum(1 for _, v in validated if v == "Correct")
    console.print(f"\n  [bold]Summary:[/bold] {correct_count}/{len(validated)} documents classified as Correct.")
    console.print(f"  [dim]Threshold for using local docs: {CORRECT_THRESHOLD}+ Correct.[/dim]\n")

    return validated


# ---------------------------------------------------------------------------
# Stage 3a: Generate answer from validated docs
# ---------------------------------------------------------------------------
def generate_answer_from_docs(question: str, docs: list[Document]) -> str:
    """Generate answer using only the validated (Correct) documents."""
    console.print(Panel("[bold cyan]Stage 3a: Generating answer from validated docs[/bold cyan]"))

    llm = get_llm(temperature=0.0)
    context = "\n\n".join([f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

    prompt = f"""Using ONLY the following validated context, answer the question.
Be precise and cite which document supports each part of your answer.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        console.print("[yellow]Hint: Make sure Ollama is running (ollama serve).[/yellow]")
        return "[Could not generate answer - LLM unavailable]"


# ---------------------------------------------------------------------------
# Stage 3b: Web fallback via DuckDuckGo HTML
# ---------------------------------------------------------------------------
def web_fallback(question: str) -> str | None:
    """
    Search DuckDuckGo HTML endpoint and parse results.
    Returns a string of search results, or None on failure.
    """
    console.print(Panel("[bold cyan]Stage 3b: Web Fallback (DuckDuckGo)[/bold cyan]"))

    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(question)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        console.print(f"  Fetching: {url[:80]}...")
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Parse result titles and snippets
        results = []
        for result in soup.select(".result"):
            title_tag = result.select_one("h2.result__title a")
            snippet_tag = result.select_one(".result__snippet")
            if title_tag and snippet_tag:
                results.append({
                    "title": title_tag.get_text(strip=True),
                    "snippet": snippet_tag.get_text(strip=True)
                })

        if not results:
            console.print("  [yellow]No results parsed from DuckDuckGo.[/yellow]")
            return None

        console.print(f"  Parsed {len(results)} web results:")
        for i, r in enumerate(results[:5], 1):
            console.print(f"    [{i}] {r['title']}")
            console.print(f"        {r['snippet'][:80]}...")

        # Format as context string
        context_parts = [f"[Web Result {i+1}] Title: {r['title']}\nSnippet: {r['snippet']}"
                        for i, r in enumerate(results[:5])]
        return "\n\n".join(context_parts)

    except ImportError:
        console.print("  [yellow]requests or BeautifulSoup not installed. Skipping web fallback.[/yellow]")
        return None
    except Exception as e:
        console.print(f"  [red]Web fallback failed: {e}[/red]")
        console.print("  [yellow]Degrading to 'insufficient knowledge' response.[/yellow]")
        return None


def generate_answer_from_web(question: str, web_context: str) -> str:
    """Generate answer using web search results."""
    llm = get_llm(temperature=0.0)

    prompt = f"""Using the following web search results, provide a helpful answer to the question.
Cite which web result supports each claim. Note that these are from the open web,
not from our local knowledge base.

Web Results:
{web_context}

Question: {question}

Answer:"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        return "[Could not generate answer from web results]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.print(Panel("[bold magenta]CRAG - CORRECTIVE RAG[/bold magenta]",
                        subtitle="Validate retrieval quality with web fallback"))

    # Use a question that the corpus likely answers well
    question = "What is RAG and how does it work?"

    console.print(f"\n[bold]User Question:[/bold] {question}\n")

    # Stage 0: Load corpus
    documents = load_corpus()
    vector_store = build_vector_store(documents)

    # Stage 1: Retrieve
    retrieved_docs = retrieve_documents(vector_store, question)

    # Stage 2: Validate
    validated = validate_all_documents(question, retrieved_docs)

    # Decision: use local docs or go to web?
    correct_docs = [doc for doc, verdict in validated if verdict == "Correct"]
    correct_count = len(correct_docs)

    if correct_count >= CORRECT_THRESHOLD:
        console.print(f"\n  [green]Decision: {correct_count} docs are Correct. "
                      f"Using local corpus.[/green]\n")
        final_answer = generate_answer_from_docs(question, correct_docs)
        source_label = "Local Corpus (validated)"
    else:
        console.print(f"\n  [yellow]Decision: Only {correct_count} docs are Correct "
                      f"(need {CORRECT_THRESHOLD}). Triggering web fallback.[/yellow]\n")
        web_context = web_fallback(question)
        if web_context:
            final_answer = generate_answer_from_web(question, web_context)
            source_label = "Web Search (DuckDuckGo)"
        else:
            final_answer = ("I'm sorry, I don't have sufficient knowledge to answer this question "
                           "accurately. The local corpus doesn't contain relevant information, "
                           "and the web search was unavailable.")
            source_label = "Insufficient Knowledge (degraded)"

    # Print final answer
    console.print(Panel(final_answer, title=f"[green]Final Answer[/green] (Source: {source_label})",
                        border_style="green"))

    console.print("\n[bold green]Done.[/bold green] CRAG pipeline complete.\n")


if __name__ == "__main__":
    main()
