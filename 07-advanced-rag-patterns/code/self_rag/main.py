"""
Self-RAG (Simplified)
======================
Pattern: Generate answer with citations -> Reflection pass critiques answer vs evidence
         -> If critique fails, regenerate ONCE with critique fed back.

Problem solved: LLMs can generate plausible but unsupported answers. Even with good
retrieval, the answer might overstate, omit details, or contradict evidence. Self-RAG
adds a self-reflection loop where the model critiques its own output.

This is a simplified version of the Self-RAG paper (Asai et al., 2023) which trains
special reflection tokens. Here we approximate it with a second LLM call.

Run from project root:
    uv run python 07-advanced-rag-patterns/code/self_rag/main.py
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
# Stage 1: Retrieve relevant chunks
# ---------------------------------------------------------------------------
def retrieve_chunks(vector_store: Chroma, question: str) -> list[Document]:
    """Retrieve top-K relevant chunks."""
    console.print(Panel("[bold cyan]Stage 1: Retrieving evidence[/bold cyan]"))

    results = vector_store.similarity_search(question, k=TOP_K)
    for i, doc in enumerate(results, 1):
        snippet = doc.page_content[:60] + "..." if len(doc.page_content) > 60 else doc.page_content
        console.print(f"  [{i}] {snippet}")

    return results


# ---------------------------------------------------------------------------
# Stage 2: Generate initial answer WITH citations
# ---------------------------------------------------------------------------
def generate_answer_with_citations(question: str, docs: list[Document]) -> str:
    """Generate an initial answer that cites specific chunks."""
    console.print(Panel("[bold cyan]Stage 2: Generating initial answer with citations[/bold cyan]"))

    llm = get_llm(temperature=0.0)

    # Number the chunks for citation reference
    context = "\n\n".join([f"[Chunk {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

    prompt = f"""Answer the question using ONLY the provided context.
You MUST cite your sources using [Chunk N] format after each claim.
Example: "RAG uses vector search [Chunk 1] to find relevant passages [Chunk 3]."

If the context doesn't fully answer the question, state what is missing.

Context:
{context}

Question: {question}

Answer (with [Chunk N] citations):"""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        console.print(f"\n  [bold]Initial Answer:[/bold]")
        console.print(Panel(answer, border_style="blue", width=80))
        return answer
    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        console.print("[yellow]Hint: Make sure Ollama is running (ollama serve).[/yellow]")
        return "[Generation failed]"


# ---------------------------------------------------------------------------
# Stage 3: REFLECTION pass - critique answer vs evidence
# ---------------------------------------------------------------------------
def reflect_on_answer(question: str, answer: str, docs: list[Document]) -> dict:
    """
    Ask the LLM to critique the answer against the retrieved evidence.
    
    Returns a dict with:
        - supported: bool (are claims backed by evidence?)
        - relevant: bool (does it address the question?)
        - complete: bool (are major points covered?)
        - issues: str (description of problems found)
        - pass: bool (overall pass/fail)
    """
    console.print(Panel("[bold cyan]Stage 3: REFLECTION - Critiquing answer vs evidence[/bold cyan]"))

    llm = get_llm(temperature=0.0)

    context = "\n\n".join([f"[Chunk {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

    prompt = f"""You are a critical reviewer. Evaluate the ANSWER below against the EVIDENCE.

EVIDENCE (retrieved documents):
{context}

QUESTION: {question}

ANSWER TO REVIEW:
{answer}

Evaluate on THREE criteria:
1. SUPPORTED: Is every factual claim in the answer backed by a specific chunk? 
   Are there any claims NOT found in the evidence?
2. RELEVANT: Does the answer actually address the question asked?
3. COMPLETE: Does the answer cover all major points available in the evidence?

For each criterion, respond with PASS or FAIL and a one-sentence justification.
Then give an overall verdict: PASS or FAIL.

Format your response EXACTLY as:
SUPPORTED: PASS or FAIL - <justification>
RELEVANT: PASS or FAIL - <justification>
COMPLETE: PASS or FAIL - <justification>
OVERALL: PASS or FAIL
ISSUES: <brief description of any problems, or "None">"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        console.print(f"\n  [bold]Reflection Output:[/bold]")
        console.print(Panel(raw, border_style="yellow", width=80))

        # Parse the structured response
        result = {"supported": True, "relevant": True, "complete": True,
                  "issues": "", "pass": True}

        for line in raw.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("SUPPORTED:"):
                result["supported"] = "PASS" in line_stripped.upper()
            elif line_stripped.startswith("RELEVANT:"):
                result["relevant"] = "PASS" in line_stripped.upper()
            elif line_stripped.startswith("COMPLETE:"):
                result["complete"] = "PASS" in line_stripped.upper()
            elif line_stripped.startswith("OVERALL:"):
                result["pass"] = "PASS" in line_stripped.upper()
            elif line_stripped.startswith("ISSUES:"):
                result["issues"] = line_stripped[len("ISSUES:"):].strip()

        # Print summary
        table = Table(title="Reflection Verdicts", show_lines=False)
        table.add_column("Criterion", style="bold")
        table.add_column("Result", width=8)
        table.add_row("Supported by evidence", "[green]PASS[/green]" if result["supported"] else "[red]FAIL[/red]")
        table.add_row("Relevant to question", "[green]PASS[/green]" if result["relevant"] else "[red]FAIL[/red]")
        table.add_row("Complete coverage", "[green]PASS[/green]" if result["complete"] else "[red]FAIL[/red]")
        table.add_row("[bold]OVERALL[/bold]", "[green]PASS[/green]" if result["pass"] else "[red]FAIL[/red]")
        console.print(table)

        if result["issues"] and result["issues"] != "None":
            console.print(f"\n  [yellow]Issues identified:[/yellow] {result['issues']}")

        return result

    except Exception as e:
        console.print(f"[red]Reflection LLM call failed: {e}[/red]")
        console.print("[yellow]Hint: Make sure Ollama is running (ollama serve).[/yellow]")
        # On failure, assume pass (don't block the pipeline)
        return {"supported": True, "relevant": True, "complete": True,
                "issues": "Reflection failed", "pass": True}


# ---------------------------------------------------------------------------
# Stage 4: Regenerate with critique (if needed)
# ---------------------------------------------------------------------------
def regenerate_with_critique(question: str, docs: list[Document],
                              previous_answer: str, critique: dict) -> str:
    """Regenerate the answer, feeding back the critique as additional guidance."""
    console.print(Panel("[bold cyan]Stage 4: Regenerating with critique feedback[/bold cyan]"))

    llm = get_llm(temperature=0.0)

    context = "\n\n".join([f"[Chunk {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

    issues_text = critique.get("issues", "The previous answer had issues.")
    if not critique["supported"]:
        issues_text += " Specifically: some claims were not supported by the cited chunks."
    if not critique["complete"]:
        issues_text += " Specifically: major points from the evidence were omitted."
    if not critique["relevant"]:
        issues_text += " Specifically: the answer drifted from the actual question."

    prompt = f"""You previously generated an answer that was CRITICIZED. Fix the issues.

PREVIOUS ANSWER (had problems):
{previous_answer}

CRITIQUE ISSUES:
{issues_text}

EVIDENCE:
{context}

QUESTION: {question}

Generate a CORRECTED answer that:
- Only makes claims directly supported by the evidence
- Cites [Chunk N] after each claim
- Addresses ALL major points in the evidence
- Stays focused on the question

Corrected Answer:"""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        console.print(f"\n  [bold]Corrected Answer:[/bold]")
        console.print(Panel(answer, border_style="green", width=80))
        return answer
    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        console.print("[yellow]Hint: Make sure Ollama is running (ollama serve).[/yellow]")
        return previous_answer  # Return original if regeneration fails


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.print(Panel("[bold magenta]SELF-RAG (SIMPLIFIED)[/bold magenta]",
                        subtitle="Generate -> Reflect -> Regenerate if needed"))

    question = "What are the key components of a RAG system and how do they interact?"

    console.print(f"\n[bold]User Question:[/bold] {question}\n")

    # Stage 0: Load corpus
    documents = load_corpus()
    vector_store = build_vector_store(documents)

    # Stage 1: Retrieve
    chunks = retrieve_chunks(vector_store, question)

    # Stage 2: Generate initial answer with citations
    initial_answer = generate_answer_with_citations(question, chunks)

    # Stage 3: Reflection pass
    critique = reflect_on_answer(question, initial_answer, chunks)

    # Stage 4: Decide whether to regenerate
    if critique["pass"]:
        console.print("\n  [green]Reflection PASSED. Using initial answer.[/green]\n")
        final_answer = initial_answer
    else:
        console.print("\n  [red]Reflection FAILED. Regenerating with critique feedback...[/red]\n")
        final_answer = regenerate_with_critique(question, chunks, initial_answer, critique)

    # Final output
    console.print(Panel(final_answer, title="[green]Final Answer (Self-RAG)[/green]",
                        border_style="green"))

    # Summary
    console.print("\n[bold]Pipeline Summary:[/bold]")
    console.print(f"  - Retrieved: {len(chunks)} chunks")
    console.print(f"  - Initial generation: completed")
    console.print(f"  - Reflection: {'PASSED' if critique['pass'] else 'FAILED -> regenerated'}")
    console.print(f"  - Total LLM calls: {3 if critique['pass'] else 4}")

    console.print("\n[bold green]Done.[/bold green] Self-RAG pipeline complete.\n")


if __name__ == "__main__":
    main()
