"""
GraphRAG (Lightweight)
=======================
Pattern: LLM extracts entities+relations from chunks -> build dict-of-dicts graph
         -> find communities via BFS -> LLM summarizes each community
         -> answer GLOBAL question from community summaries.

Problem solved: Basic RAG answers local questions well but fails at global questions
like "What are the main themes across ALL documents?" No single chunk contains the
global picture. GraphRAG builds a knowledge graph, finds communities, and answers
from summaries rather than raw chunks.

No external graph libraries needed - pure Python BFS for connected components.

Run from project root:
    uv run python 07-advanced-rag-patterns/code/graph_rag/main.py
"""

import sys
import pathlib
import json
import re
from collections import deque

# Add project root to path so we can import shared.config
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

# Windows console is cp1252; force UTF-8 output to avoid encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.config import get_llm, get_embeddings
from langchain_core.documents import Document

console = Console()


# ---------------------------------------------------------------------------
# Stage 0: Load corpus
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

    console.print(f"  Loaded {len(documents)} chunks from {data_dir.name}/")
    return documents


# ---------------------------------------------------------------------------
# Stage 1: Entity and Relation Extraction (per chunk)
# ---------------------------------------------------------------------------
def extract_entities_and_relations(chunk: Document, chunk_idx: int) -> tuple[list[str], list[dict]]:
    """
    Ask the LLM to extract entities and relations from a single chunk.
    
    Returns:
        entities: list of entity names
        relations: list of {"from": str, "to": str, "relation": str}
    """
    llm = get_llm(temperature=0.0)

    prompt = f"""Extract all notable ENTITIES and RELATIONS from the following text.

Text:
{chunk.page_content}

Return your answer as a JSON object with this EXACT structure:
{{
  "entities": ["Entity1", "Entity2", ...],
  "relations": [
    {{"from": "EntityA", "to": "EntityB", "relation": "describes/uses/contains/etc"}},
    ...
  ]
}}

Rules:
- Entities should be specific nouns (technologies, concepts, people, systems).
- Relations should be short verb phrases (e.g., "uses", "is part of", "enables").
- Only include entities that actually appear or are clearly implied in the text.
- Maximum 10 entities, maximum 15 relations.
- Output ONLY valid JSON. No markdown, no explanation."""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Try to parse JSON directly
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from the response
            # Look for the first { ... } block
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                console.print(f"  [yellow]Chunk {chunk_idx}: Could not parse JSON. Skipping.[/yellow]")
                return [], []

        entities = data.get("entities", [])
        relations = data.get("relations", [])

        # Validate structure
        valid_entities = [e for e in entities if isinstance(e, str) and e.strip()]
        valid_relations = []
        for r in relations:
            if isinstance(r, dict) and "from" in r and "to" in r and "relation" in r:
                valid_relations.append({
                    "from": str(r["from"]).strip(),
                    "to": str(r["to"]).strip(),
                    "relation": str(r["relation"]).strip()
                })

        console.print(f"  Chunk {chunk_idx}: {len(valid_entities)} entities, "
                      f"{len(valid_relations)} relations extracted.")
        return valid_entities, valid_relations

    except Exception as e:
        console.print(f"  [red]Chunk {chunk_idx}: LLM call failed: {e}[/red]")
        console.print("  [yellow]Hint: Make sure Ollama is running (ollama serve).[/yellow]")
        return [], []


# ---------------------------------------------------------------------------
# Stage 2: Build the knowledge graph (dict of dicts)
# ---------------------------------------------------------------------------
def build_graph(all_extractions: list[tuple[list[str], list[dict]]]) -> dict[str, dict[str, str]]:
    """
    Merge all entity/relation extractions into a single undirected graph.
    
    Graph representation: {entity: {related_entity: relation_string}}
    Undirected: if A->B exists, B->A also exists.
    """
    console.print(Panel("[bold cyan]Stage 2: Building knowledge graph[/bold cyan]"))

    graph: dict[str, dict[str, str]] = {}

    for entities, relations in all_extractions:
        # Ensure all entities are nodes (even if they have no relations yet)
        for entity in entities:
            if entity not in graph:
                graph[entity] = {}

        # Add relations (undirected)
        for rel in relations:
            src, dst, rel_str = rel["from"], rel["to"], rel["relation"]
            if src not in graph:
                graph[src] = {}
            if dst not in graph:
                graph[dst] = {}
            # Add both directions (undirected graph)
            graph[src][dst] = rel_str
            graph[dst][src] = f"related to ({rel_str})"

    num_nodes = len(graph)
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2  # Undirected

    console.print(f"  Graph built: {num_nodes} nodes, {num_edges} edges.")

    # Print the graph
    console.print(f"\n  [bold]Graph Nodes ({num_nodes}):[/bold]")
    for i, node in enumerate(sorted(graph.keys()), 1):
        console.print(f"    {i}. {node}")

    console.print(f"\n  [bold]Graph Edges ({num_edges}):[/bold]")
    printed_edges = set()
    edge_count = 0
    for src, neighbors in sorted(graph.items()):
        for dst, rel in sorted(neighbors.items()):
            # Avoid printing both directions
            edge_key = tuple(sorted([src, dst]))
            if edge_key not in printed_edges:
                printed_edges.add(edge_key)
                console.print(f"    {src} --[{rel}]--> {dst}")
                edge_count += 1
                if edge_count >= 20:  # Limit output
                    console.print("    ... (truncated)")
                    break
        if edge_count >= 20:
            break

    return graph


# ---------------------------------------------------------------------------
# Stage 3: Find communities via BFS (connected components)
# ---------------------------------------------------------------------------
def find_communities_bfs(graph: dict[str, dict[str, str]]) -> list[list[str]]:
    """
    Find connected components using BFS. Pure Python, no external libs.
    
    Each connected component is a "community" - a cluster of entities that are
    related to each other but not to entities in other clusters.
    """
    console.print(Panel("[bold cyan]Stage 3: Finding communities (BFS connected components)[/bold cyan]"))

    visited: set[str] = set()
    communities: list[list[str]] = []

    for node in graph:
        if node in visited:
            continue

        # BFS from this node
        queue = deque([node])
        visited.add(node)
        component = []

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbor in graph.get(current, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        communities.append(component)

    # Sort by size descending
    communities.sort(key=len, reverse=True)

    console.print(f"  Found {len(communities)} communities:")
    for i, comm in enumerate(communities, 1):
        members = ", ".join(comm[:5])
        if len(comm) > 5:
            members += f" (+{len(comm)-5} more)"
        console.print(f"    Community {i} ({len(comm)} nodes): {members}")

    return communities


# ---------------------------------------------------------------------------
# Stage 4: Summarize each community
# ---------------------------------------------------------------------------
def summarize_community(community: list[str], graph: dict[str, dict[str, str]],
                         community_idx: int) -> str:
    """Ask the LLM to summarize what a community of entities is about."""
    llm = get_llm(temperature=0.0)

    # Build a description of the community's internal structure
    internal_edges = []
    for entity in community:
        for neighbor, rel in graph.get(entity, {}).items():
            if neighbor in community:  # Only internal edges
                internal_edges.append(f"  - {entity} --[{rel}]--> {neighbor}")

    edges_text = "\n".join(internal_edges[:15])  # Cap at 15 edges for prompt size

    prompt = f"""You are analyzing a community of related concepts from a knowledge graph.

Community members: {', '.join(community)}

Internal relationships:
{edges_text}

Write a 2-3 sentence summary explaining what this community represents,
what theme it covers, and how its members relate to each other.
Be specific and informative. No preamble."""

    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()
        console.print(f"  Community {community_idx} summary: {summary[:80]}...")
        return summary
    except Exception as e:
        console.print(f"  [red]Community {community_idx}: LLM call failed: {e}[/red]")
        return f"[Summary unavailable - LLM error]"


# ---------------------------------------------------------------------------
# Stage 5: Answer GLOBAL question from community summaries
# ---------------------------------------------------------------------------
def answer_global_question(question: str, community_summaries: list[tuple[list[str], str]]) -> str:
    """
    Answer a global question using ONLY the community summaries (not raw chunks).
    This is the key advantage of GraphRAG: global questions answered from
    structured summaries rather than individual chunks.
    """
    console.print(Panel("[bold cyan]Stage 5: Answering global question from community summaries[/bold cyan]"))

    llm = get_llm(temperature=0.0)

    # Build context from community summaries
    context_parts = []
    for i, (members, summary) in enumerate(community_summaries, 1):
        context_parts.append(f"Community {i} (entities: {', '.join(members[:5])}):\n{summary}")

    context = "\n\n".join(context_parts)

    prompt = f"""You have access to a knowledge graph that was built from a collection of documents.
The graph has been decomposed into communities, each with a summary.

COMMUNITY SUMMARIES:
{context}

GLOBAL QUESTION: {question}

Using ONLY the community summaries above (not raw text), provide a comprehensive answer
that synthesizes information across ALL communities. Highlight how different themes
connect to each other.

Answer:"""

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        return answer
    except Exception as e:
        console.print(f"[red]LLM call failed: {e}[/red]")
        console.print("[yellow]Hint: Make sure Ollama is running (ollama serve).[/yellow]")
        return "[Could not generate answer - LLM unavailable]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.print(Panel("[bold magenta]GRAPH RAG (LIGHTWEIGHT)[/bold magenta]",
                        subtitle="Entity extraction -> Graph -> Communities -> Global answers"))

    question = "What are the main themes across all documents?"

    console.print(f"\n[bold]Global Question:[/bold] {question}\n")

    # Stage 0: Load corpus
    documents = load_corpus()

    # Stage 1: Extract entities and relations from each chunk
    console.print(Panel("[bold cyan]Stage 1: Entity & Relation Extraction[/bold cyan]"))
    all_extractions = []
    for i, doc in enumerate(documents):
        entities, relations = extract_entities_and_relations(doc, i + 1)
        all_extractions.append((entities, relations))

    # Stage 2: Build the graph
    graph = build_graph(all_extractions)

    # Stage 3: Find communities via BFS
    communities = find_communities_bfs(graph)

    # Stage 4: Summarize each community
    console.print(Panel("[bold cyan]Stage 4: Community Summarization[/bold cyan]"))
    community_summaries = []
    for i, community in enumerate(communities, 1):
        summary = summarize_community(community, graph, i)
        community_summaries.append((community, summary))

    # Print all summaries
    table = Table(title="Community Summaries", show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Members", width=40)
    table.add_column("Summary", width=60)
    for i, (members, summary) in enumerate(community_summaries, 1):
        members_str = ", ".join(members[:4])
        if len(members) > 4:
            members_str += f" (+{len(members)-4})"
        table.add_row(str(i), members_str, summary[:80] + ("..." if len(summary) > 80 else ""))
    console.print(table)

    # Stage 5: Answer the global question
    final_answer = answer_global_question(question, community_summaries)

    console.print(Panel(final_answer, title="[green]Global Answer (from community summaries)[/green]",
                        border_style="green"))

    # Summary
    console.print("\n[bold]Pipeline Summary:[/bold]")
    console.print(f"  - Chunks processed: {len(documents)}")
    console.print(f"  - Graph nodes: {len(graph)}")
    console.print(f"  - Communities found: {len(communities)}")
    console.print(f"  - LLM calls: {len(documents)} (extraction) + {len(communities)} (summaries) + 1 (answer)")

    console.print("\n[bold green]Done.[/bold green] GraphRAG pipeline complete.\n")


if __name__ == "__main__":
    main()
