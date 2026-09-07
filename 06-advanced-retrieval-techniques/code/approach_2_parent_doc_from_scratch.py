"""
Module 6 · Approach 2 — Parent-Document Retriever FROM SCRATCH (no LangChain)
============================================================================
Re-implements the classic "small-to-big" pattern with zero LangChain
retriever code — just plain Python + raw ChromaDB:

  1. Split each source doc into LARGE PARENTS  (~1500 chars)
  2. Split each parent into SMALL CHILDREN     (~300 chars)
  3. Index ONLY the children in raw chroma, tagging each with its parent_id
  4. At query time: embed the question -> top-k CHILDREN
  5. Map children back to their parents, dedupe (preserving rank order)
  6. Return the parent texts as the final context

Why bother? The LangChain ParentDocumentRetriever hides this whole flow.
Building it by hand shows exactly where the cost lives (ingestion-time
splitting + one KV lookup per hit) and why it needs NO extra LLM calls.

No LLM required — pure vector search. Always runs.

Run from the project root:
    uv run python 06-advanced-retrieval-techniques/code/approach_2_parent_doc_from_scratch.py
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib
import re
from dataclasses import dataclass, field
from pathlib import Path

# Make `shared.config` importable no matter which directory we launch from.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_embeddings

import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

PARENT_SIZE = 1500   # large context window returned to the generator
CHILD_SIZE = 300     # small window used for precise matching
TOP_K_CHILDREN = 5


# ---------------------------------------------------------------------------
# 1. A TINY SPLITTER (sentence-aware greedy packing — no LangChain)
# ---------------------------------------------------------------------------
def split_text(text: str, max_chars: int) -> list[str]:
    """Greedy sentence packing: never cut mid-sentence unless a single
    sentence is longer than max_chars (then hard-cut it)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        # Hard-cut pathologically long sentences so we always respect max_chars.
        while len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sent[:max_chars])
            sent = sent[max_chars:].strip()
        if not sent:
            continue
        if current and len(current) + 1 + len(sent) > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks


@dataclass
class ChildHit:
    """One retrieved child chunk plus the link back to its parent."""
    text: str
    distance: float          # cosine distance from the query (lower = closer)
    parent_id: str
    source: str


@dataclass
class ParentDoc:
    id: str
    source: str
    text: str
    child_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. BUILD THE PARENT/CHILD INDEX (ingestion time — the only real cost)
# ---------------------------------------------------------------------------
def build_index() -> tuple[dict[str, ParentDoc], chromadb.Collection]:
    """Read every sample file, split into parents -> children, index children."""
    parents: dict[str, ParentDoc] = {}
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        "m6_scratch_children", metadata={"hnsw:space": "cosine"})

    n_children = 0
    for path in sorted(SAMPLES_DIR.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        for pi, parent_text in enumerate(split_text(text, PARENT_SIZE)):
            pid = f"{path.name}::p{pi}"
            parents[pid] = ParentDoc(id=pid, source=path.name, text=parent_text)
            for ci, child_text in enumerate(split_text(parent_text, CHILD_SIZE)):
                cid = f"{pid}::c{ci}"
                parents[pid].child_ids.append(cid)
                col.add(
                    ids=[cid],
                    documents=[child_text],
                    metadatas=[{"parent_id": pid, "source": path.name}],
                )
                n_children += 1
    return parents, col


# ---------------------------------------------------------------------------
# 3. RETRIEVE: children in, parents out
# ---------------------------------------------------------------------------
def retrieve(col: chromadb.Collection, parents: dict[str, ParentDoc],
             question: str, embeddings) -> tuple[list[ChildHit], list[ParentDoc]]:
    """Embed the question, fetch top-k children, map back to unique parents."""
    q_vec = embeddings.embed_query(question)
    res = col.query(query_embeddings=[q_vec], n_results=TOP_K_CHILDREN,
                    include=["documents", "metadatas", "distances"])
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    hits = [ChildHit(text=docs[i], distance=dists[i],
                     parent_id=metas[i]["parent_id"], source=metas[i]["source"])
            for i in range(len(ids))]

    # Dedupe parents, preserving the rank order of their best child.
    seen: set[str] = set()
    out: list[ParentDoc] = []
    for h in hits:
        if h.parent_id not in seen:
            seen.add(h.parent_id)
            out.append(parents[h.parent_id])
    return hits, out


# ---------------------------------------------------------------------------
# 4. PRETTY PRINTING
# ---------------------------------------------------------------------------
def show_child_hits(hits: list[ChildHit]) -> None:
    table = Table(title="Child hits (what was actually matched)",
                  title_style="bold cyan", header_style="bold magenta")
    table.add_column("#", width=3, justify="right", style="dim")
    table.add_column("dist", width=7)
    table.add_column("source", width=22)
    table.add_column("parent_id", width=34)
    table.add_column("child snippet", overflow="fold")
    for i, h in enumerate(hits, 1):
        table.add_row(str(i), f"{h.distance:.4f}", h.source, h.parent_id,
                      " ".join(h.text.split())[:120])
    console.print(table)


def show_parents(parents: list[ParentDoc]) -> None:
    for i, p in enumerate(parents, 1):
        console.print(Panel(
            f"[bold cyan]Parent {i}[/bold cyan]  [dim]{p.id} · {len(p.text)} chars · "
            f"{len(p.child_ids)} children[/dim]\n\n"
            + " ".join(p.text.split())[:600]
            + ("…" if len(p.text) > 600 else ""),
            border_style="green"))


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    console.rule("[bold]MODULE 6 · PARENT-DOCUMENT RETRIEVER — FROM SCRATCH")
    console.print(f"[dim]Parents ≈{PARENT_SIZE} chars · children ≈{CHILD_SIZE} chars · "
                  f"top-{TOP_K_CHILDREN} children per query · 0 LLM calls[/dim]\n")

    try:
        parents, col = build_index()
    except Exception as e:  # noqa: BLE001 - friendly top-level handler
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        console.print("[yellow]⚠ This step failed (not an LLM issue). Check that "
                      "`uv sync` was run and that data/samples/ exists.[/yellow]")
        return

    n_children_total = sum(len(p.child_ids) for p in parents.values())
    console.print(f"[green]Indexed {len(parents)} parents / {n_children_total} children "
                  f"in raw ChromaDB (in-memory).[/green]\n")

    embeddings = get_embeddings()
    questions = [
        "Explain how HNSW builds its multi-layer graph.",
        "What products does Acme Robotics sell?",
    ]
    for q in questions:
        console.print(f"\n[bold]Question:[/bold] {q}")
        try:
            hits, result = retrieve(col, parents, q, embeddings)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]{type(e).__name__}: {e}[/red]")
            console.print("[yellow]⚠ Retrieval failed (not an LLM issue).[/yellow]")
            continue

        show_child_hits(hits)
        console.print(f"\n[bold green]{len(result)} unique parent(s)[/bold green] "
                      f"recovered from {len(hits)} child hits:\n")
        show_parents(result)

    console.print("\n[dim]Observation: tiny children (≈300 chars) give precise matching, but the "
                  "generator receives the larger parent (≈1500 chars) so surrounding context is "
                  "never lost. All of this happens with zero LLM calls — the only cost is the "
                  "extra splitting work at ingestion.[/dim]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    except Exception as exc:  # noqa: BLE001 - friendly top-level handler
        console.print(Panel(f"[red]{type(exc).__name__}: {exc}[/red]", title="Startup error"))
        console.print("[dim]Make sure dependencies are installed (`uv sync`) and that the local "
                      "embedding model (all-MiniLM-L6-v2) is available.[/dim]")
