"""
Module 4 - Vector Stores: Approach B - RAW ChromaDB API (NO LangChain).

Run from the project root:
    uv run python 04-vector-stores/code/approach_2_raw_chroma_api.py

Why this exists:
    Approach A (main.py) also uses raw chromadb, but this script goes further:
    it treats chromadb as a *standalone vector database* - the way you would
    in a service that does NOT depend on LangChain at all. Every call below
    is a plain `client.*` / `collection.*` method. If you understand this
    file, you can drive ANY vector DB (Qdrant, pgvector, Milvus) because
    they all expose the same CRUD surface.

Covers (each operation's result is printed):
    1. CREATE   - PersistentClient + create_collection (cosine space)
    2. ADD      - add() with ids + documents + metadatas
    3. READ     - query() with n_results, get() by ids / where
    4. UPDATE   - upsert() replacing content + metadata in place
    5. DELETE   - delete() by ids AND by where-clause
    6. COUNT    - count() after every mutation
    7. LIFECYCLE- list_collections(), delete_collection()

Persistence dir: a TEMP dir under data/ (data/chroma_raw_api) so this lab
never touches the main CHROMA_DIR used by other modules. It is wiped at
start and the collection deleted at the end.

NO LLM / NO API keys - only the local MiniLM embedding model + ChromaDB.
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib
import shutil

# Make the project root importable so we can reuse shared.config from any folder.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from shared.config import get_embeddings

console = Console()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMP_DIR = PROJECT_ROOT / "data" / "chroma_raw_api"   # temp persist dir under data/
COLLECTION_NAME = "raw_api_lab"


class _ChromaEF:
    """Adapter: expose a LangChain embeddings object as a ChromaDB EmbeddingFunction.

    ChromaDB expects a callable `List[str] -> List[List[float]]`. Wrapping our
    shared embedding model guarantees ADD and QUERY use the SAME model, so
    distances are meaningful. (This is the only 'LangChain' object in play -
    everything else is raw chromadb.)
    """
    def __init__(self, lc):
        self._lc = lc

    def __call__(self, input):
        return self._lc.embed_documents(list(input))

    def embed_query(self, input):
        # Newer ChromaDB versions call embed_query() explicitly on the query path,
        # passing a LIST of texts. LangChain's embed_query() takes ONE string.
        items = list(input)
        if len(items) == 1:
            return [self._lc.embed_query(items[0])]
        return self._lc.embed_documents(items)


def collection_names(client):
    """Robustly list collection names across chromadb versions (objects or strings)."""
    return [getattr(c, "name", str(c)) for c in client.list_collections()]


def show_result(title: str, res: dict, max_snippet: int = 60) -> None:
    """Pretty-print a chromadb query result (ids + distances + documents)."""
    ids = res["ids"][0]
    dists = res["distances"][0] if "distances" in res else [None] * len(ids)
    docs = res["documents"][0] if "documents" in res else [""] * len(ids)
    t = Table(title=title, title_style="bold cyan")
    t.add_column("id")
    t.add_column("distance", justify="right")
    t.add_column("snippet")
    for rid, dist, doc in zip(ids, dists, docs):
        d = f"{dist:.3f}" if dist is not None else "-"
        t.add_row(rid, d, (doc or "")[:max_snippet].replace("\n", " ") + ("..." if len(doc or "") > max_snippet else ""))
    console.print(t)


def main() -> None:
    console.rule("[bold cyan]Module 4 - Approach B: Raw ChromaDB API (no LangChain)[/]")

    # ---- Setup: embeddings + TEMP persistent client -----------------------
    try:
        emb = get_embeddings()
    except Exception as exc:
        console.print(Panel(
            f"[red]Embedding model failed to load:[/]\n{exc}\n\n"
            "[bold]Hint:[/bold] this script needs the local MiniLM model. Make sure "
            "sentence-transformers can reach Hugging Face once (it caches afterwards), "
            "or set EMBEDDING_PROVIDER in .env.",
            title="Setup issue", border_style="red"))
        sys.exit(0)

    import chromadb

    # Fresh temp persist dir every run -> fully reproducible lab.
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    client = chromadb.PersistentClient(path=str(TEMP_DIR))
    console.print(f"\n[bold]Persistent client[/bold] -> {TEMP_DIR}")

    # ---- 1. CREATE ---------------------------------------------------------
    console.print("\n[bold]1. CREATE[/bold] collection (cosine space)")
    # Under the hood: an empty HNSW graph is allocated for this collection.
    col = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_ChromaEF(emb),
        metadata={"hnsw:space": "cosine"},
    )
    console.print(f"  created '{col.name}'  count={col.count()}  metadata={col.metadata}")

    # ---- 2. ADD -------------------------------------------------------------
    console.print("\n[bold]2. ADD[/bold] 5 docs with structured metadata")
    docs = [
        ("doc_1", "ChromaDB uses an HNSW graph index for approximate nearest neighbor search.",
         {"source": "notes.md", "page": 1, "tags": ["indexing"], "score": 9}),
        ("doc_2", "Cosine distance compares direction, ignoring vector magnitude.",
         {"source": "notes.md", "page": 2, "tags": ["metrics"], "score": 7}),
        ("doc_3", "Metadata filters are applied before the ANN search walks the graph.",
         {"source": "guide.pdf", "page": 10, "tags": ["filtering", "indexing"], "score": 8}),
        ("doc_4", "Upsert replaces an existing id's vector and metadata in place.",
         {"source": "guide.pdf", "page": 12, "tags": ["crud"], "score": 6}),
        ("doc_5", "PersistentClient keeps collections on disk across process restarts.",
         {"source": "guide.pdf", "page": 15, "tags": ["persistence"], "score": 10}),
    ]
    col.add(ids=[d[0] for d in docs],
            documents=[d[1] for d in docs],
            metadatas=[d[2] for d in docs])
    console.print(f"  added {len(docs)} docs  count={col.count()}")

    # ---- 3. READ ------------------------------------------------------------
    console.print("\n[bold]3. READ[/bold] - similarity query with n_results=3")
    res = col.query(query_texts=["How does the ANN index work?"], n_results=3)
    show_result("Top-3 for 'How does the ANN index work?'", res)

    console.print("\n[bold]3b. READ[/bold] - get() by ids and by where-filter")
    got = col.get(ids=["doc_2"])
    console.print(f"  get(ids=['doc_2'])      -> {got['documents'][0][:60]}...")
    got_f = col.get(where={"source": "guide.pdf"})
    console.print(f"  get(where source=pdf)  -> {sorted(got_f['ids'])}")

    # ---- 4. UPDATE (upsert) --------------------------------------------------
    console.print("\n[bold]4. UPDATE[/bold] - upsert doc_3 with new text + metadata")
    col.upsert(ids=["doc_3"],
               documents=["Pre-filtering prunes candidates BEFORE the HNSW beam search."],
               metadatas=[{"source": "guide.pdf", "page": 11, "tags": ["filtering"], "score": 9}])
    after = col.get(ids=["doc_3"])
    console.print(f"  doc_3 now reads : {after['documents'][0]}")
    console.print(f"  doc_3 metadata  : {after['metadatas'][0]}")
    console.print(f"  count unchanged : {col.count()} (upsert replaced, did not insert)")

    # ---- 5. DELETE -----------------------------------------------------------
    console.print("\n[bold]5. DELETE[/bold] - by ids, then by where-clause")
    before = col.count()
    col.delete(ids=["doc_4"])
    console.print(f"  delete(ids=['doc_4'])     count {before} -> {col.count()}")
    before = col.count()
    col.delete(where={"source": "guide.pdf"})        # delete ALL guide.pdf docs at once
    console.print(f"  delete(where source=pdf) count {before} -> {col.count()}")
    remaining = col.get(include=["metadatas"])
    console.print(f"  remaining ids: {sorted(remaining['ids'])}")

    # ---- 6. COUNT --------------------------------------------------------------
    console.print("\n[bold]6. COUNT[/bold]")
    console.print(f"  col.count() = {col.count()}")

    # ---- 7. COLLECTION LIFECYCLE -------------------------------------------------
    console.print("\n[bold]7. LIFECYCLE[/bold] - list / delete collection")
    console.print(f"  collections before: {collection_names(client)}")
    client.delete_collection(COLLECTION_NAME)
    console.print(f"  deleted '{COLLECTION_NAME}'")
    console.print(f"  collections after : {collection_names(client)}")

    # Cleanup temp dir (leave nothing behind). Close the client first so the
    # SQLite handles are released (Windows locks open files).
    client.close()
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except Exception:
        pass

    console.rule("[bold green]Raw ChromaDB API lab complete [OK][/]")
    console.print(Panel(
        "[dim]Key takeaway: create / add / query / upsert / delete / count / "
        "delete_collection is the ENTIRE surface you need. LangChain's VectorStore "
        "is a thin wrapper over exactly these calls.[/]",
        title="Under the hood", border_style="blue"))


if __name__ == "__main__":
    main()
