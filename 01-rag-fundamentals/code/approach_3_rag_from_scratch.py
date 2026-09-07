"""Module 1 — Approach 3: RAG from scratch (NO LangChain).

Proves you understand what the framework hides. Only three building blocks:
  1. chromadb.PersistentClient  — raw vector store API (cosine space)
  2. sentence-transformers      — local MiniLM embeddings (cached, no download)
  3. requests                   — direct OpenAI-compatible /chat/completions HTTP call

LLM endpoint resolution (from .env at project root):
  OPENAI_API_KEY        -> https://api.openai.com/v1, model OPENAI_MODEL (gpt-4o-mini)
  YOLO_AUTO_API_KEY     -> YOLO_AUTO_BASE_URL (https://yolo-auto.com/v1), YOLO_AUTO_MODEL

Run from project root:
    uv run python 01-rag-fundamentals/code/approach_3_rag_from_scratch.py
"""
import sys
import pathlib
import os
import json
import hashlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CHUNK_SIZE = 400   # chars per chunk
CHUNK_OVERLAP = 60 # chars of overlap between chunks
TOP_K = 3

EXAMPLE_QUESTIONS = [
    "What is Retrieval-Augmented Generation and what problems does it solve?",
    "Which indexing strategies are mentioned for vector databases?",
]


def resolve_llm_endpoint():
    """Return (api_key, base_url, model, label) or None if no cloud key is set."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return openai_key, "https://api.openai.com/v1", os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "OpenAI"
    yolo_key = os.getenv("YOLO_AUTO_API_KEY", "").strip()
    if yolo_key:
        base = os.getenv("YOLO_AUTO_BASE_URL", "https://yolo-auto.com/v1").rstrip("/")
        return yolo_key, base, os.getenv("YOLO_AUTO_MODEL", "qwen3.8-27b"), "Yolo-Auto"
    return None


def load_documents():
    """Read every sample file as plain text — no LangChain loaders."""
    samples = PROJECT_ROOT / "data" / "samples"
    docs = []  # list of (source_name, text)
    for path in sorted(samples.iterdir()):
        if path.suffix not in {".txt", ".md", ".csv", ".json"}:
            continue
        raw = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            try:
                raw = json.dumps(json.loads(raw), indent=2)
            except json.JSONDecodeError:
                pass
        elif path.suffix == ".csv":
            raw = "\n".join(line.strip() for line in raw.splitlines() if line.strip())
        docs.append((path.name, raw))
    console.print(f"[bold green]✓ Loaded {len(docs)} documents from data/samples[/bold green]")
    return docs


def chunk_text(text: str):
    """Naive sliding-window chunker with overlap (no sentence logic on purpose)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def stable_id(source: str, index: int, text: str) -> str:
    return hashlib.sha1(f"{source}:{index}:{text[:80]}".encode("utf-8")).hexdigest()


def build_store(docs):
    """Embed all chunks with local MiniLM and upsert into a persistent Chroma collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    console.print(f"[dim]Loading embeddings model: {model_name}[/dim]")
    embedder = SentenceTransformer(model_name)

    records = []  # (id, text, metadata)
    for source, raw in docs:
        for i, chunk in enumerate(chunk_text(raw)):
            records.append((stable_id(source, i, chunk), chunk, {"source": source}))
    console.print(f"[bold green]✓ Created {len(records)} chunks[/bold green]")

    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma_db"))
    name = "module1_from_scratch"
    try:
        client.delete_collection(name)  # idempotent rebuild
    except Exception:
        pass
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=[r[0] for r in records],
        documents=[r[1] for r in records],
        metadatas=[r[2] for r in records],
        embeddings=embedder.encode([r[1] for r in records], normalize_embeddings=True).tolist(),
    )
    console.print(f"[bold green]✓ Vector store ready ({col.count()} vectors, cosine space)[/bold green]")
    return col, embedder


def chat(base_url: str, api_key: str, model: str, messages, timeout: int = 90) -> str:
    """One direct HTTP call to an OpenAI-compatible /chat/completions endpoint."""
    import requests

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.0},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def ask(col, embedder, endpoint, question: str) -> None:
    api_key, base_url, model, label = endpoint
    console.rule(f"[bold cyan]Q: {question}[/bold cyan]")

    # --- RETRIEVE: cosine similarity search via raw Chroma API ---
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=TOP_K)
    hits = list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))
    if not hits:
        console.print("[yellow]No relevant chunks found.[/yellow]\n")
        return

    table = Table(title=f"Top-{len(hits)} retrieved (cosine distance — lower = closer)", expand=True)
    table.add_column("#", justify="right")
    table.add_column("dist")
    table.add_column("source")
    table.add_column("chunk")
    for i, (doc, meta, dist) in enumerate(hits, 1):
        table.add_row(str(i), f"{dist:.3f}", meta.get("source", "?"), doc[:140].replace("\n", " ") + "…")
    console.print(table)

    # --- GENERATE: hand-assembled prompt + direct HTTP call ---
    context = "\n\n".join(
        f"[{i}] (source: {meta.get('source', '?')})\n{doc}" for i, (doc, meta, _) in enumerate(hits, 1)
    )
    messages = [
        {"role": "system", "content": (
            "You are a precise assistant. Use ONLY the numbered context below to answer. "
            "Cite sources inline like [1], [2]. If the context is insufficient, say so."
        )},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"},
    ]
    answer = chat(base_url, api_key, model, messages)
    console.print(Panel(answer, title=f"Answer ({label} · {model})", border_style="green"))
    console.print()


def main():
    console.print(Panel("[bold]Module 1 — Approach 3: RAG from Scratch (no LangChain)[/bold]", border_style="blue"))

    endpoint = resolve_llm_endpoint()
    if endpoint is None:
        console.print(
            Panel(
                "[yellow]No cloud LLM key configured.\n\n"
                "This approach calls an OpenAI-compatible API directly with `requests`,\n"
                "so it needs one of:\n"
                "  • OPENAI_API_KEY        (uses OPENAI_MODEL, default gpt-4o-mini)\n"
                "  • YOLO_AUTO_API_KEY     (uses YOLO_AUTO_BASE_URL + YOLO_AUTO_MODEL)\n\n"
                "Copy .env.example to .env, fill in a key, and re-run.[/yellow]",
                title="⚠ Graceful degradation",
                border_style="yellow",
            )
        )
        sys.exit(0)

    _, _, model, label = endpoint
    console.print(f"[bold green]✓ LLM endpoint:[/bold green] {label} ({model})")

    docs = load_documents()
    col, embedder = build_store(docs)

    import requests
    for q in EXAMPLE_QUESTIONS:
        try:
            ask(col, embedder, endpoint, q)
        except requests.RequestException as e:
            console.print(
                Panel(
                    f"[yellow]LLM request failed ({type(e).__name__}: {e}).\n\n"
                    "Check your network and that the API key in .env is valid.\n"
                    "The retrieval half above ran fine — only generation needs the cloud.[/yellow]",
                    title="⚠ Graceful degradation",
                    border_style="yellow",
                )
            )
            sys.exit(0)

    console.rule("[bold red]What LangChain was hiding[/bold red]")
    console.print(
        """
• A 'retriever' is just: embed query → ANN search → wrap results in objects.
• A 'prompt template' is just string formatting; a 'chain' is just function composition.
• The real value of frameworks: loaders/splitters for 50+ formats, caching, tracing,
  and swappable components — not the core loop, which fits in ~100 lines.
"""
    )


if __name__ == "__main__":
    main()
