"""
Module 4 - Vector Stores: full ChromaDB CRUD lab.

Run from the project root:
    uv run python 04-vector-stores/code/main.py

Covers: CREATE / ADD / READ / UPDATE(upsert) / DELETE, metadata filtering, persistence
(CHROMA_DIR), collection lifecycle, and recall-style sanity checks. NO LLM / NO API keys -
only the local embedding model + ChromaDB.

Heavily commented about what happens UNDER THE HOOD (HNSW index).

NOTE: console output is kept pure ASCII so it renders on ANY terminal/locale
      (Windows cp1252, macOS, Linux) without UnicodeEncodeError.
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib
import csv
import json

# Make the project root importable so we can reuse shared.config from any folder.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from shared.config import get_embeddings, CHROMA_DIR

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = PROJECT_ROOT / "data" / "samples"
COLLECTION_NAME = "course_docs"


class _ChromaEF:
    """Adapter: expose a LangChain embeddings object as a ChromaDB EmbeddingFunction.

    ChromaDB expects a callable `List[str] -> List[List[float]]`. LangChain's embeddings
    expose exactly that via `embed_documents`. Wrapping them guarantees that ADD and QUERY
    use the SAME model, so distances are meaningful.
    """
    def __init__(self, lc):
        self._lc = lc

    def __call__(self, input):
        return self._lc.embed_documents(list(input))

    def embed_query(self, input):
        # Newer ChromaDB versions call embed_query() explicitly on the query path,
        # passing a LIST of texts. LangChain's embed_query() takes ONE string, so:
        items = list(input)
        if len(items) == 1:
            return [self._lc.embed_query(items[0])]
        return self._lc.embed_documents(items)


def load_sample_docs():
    """Read the four sample files into (id, text, metadata) tuples."""
    docs = []

    # 1) rag_overview.txt -> a single 'article' doc.
    rag_text = (SAMPLES / "rag_overview.txt").read_text(encoding="utf-8").strip()
    docs.append(("rag_overview", rag_text,
                 {"source": "rag_overview.txt", "type": "article", "topic": "rag"}))

    # 2) products.csv -> ONE doc per row, with structured (filterable) metadata.
    with open(SAMPLES / "products.csv", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            text = (f"{row['product']} is a {row['category'].lower()} priced at "
                    f"${row['price_usd']} with {row['stock']} units in stock (region {row['region']}).")
            docs.append((
                f"product_{i}",
                text,
                {"source": "products.csv", "type": "product",
                 "category": row["category"], "region": row["region"],
                 "price_usd": float(row["price_usd"]), "stock": int(row["stock"])},
            ))

    # 3) company_profile.json -> one 'company' doc summarizing the profile.
    company = json.loads((SAMPLES / "company_profile.json").read_text(encoding="utf-8"))
    ctext = (f"{company['company']} was founded in {company['founded']} in "
             f"{company['headquarters']}. Mission: {company['mission']}. Products: "
             + ", ".join(p["name"] for p in company["products"]) + ".")
    docs.append(("company_profile", ctext,
                 {"source": "company_profile.json", "type": "company",
                  "name": company["company"]}))

    # 4) vector_db_notes.md -> one 'notes' doc (great for the similarity-search demo).
    vdb_text = (SAMPLES / "vector_db_notes.md").read_text(encoding="utf-8").strip()
    docs.append(("vector_db_notes", vdb_text,
                 {"source": "vector_db_notes.md", "type": "notes", "topic": "vector-db"}))

    return docs


def collection_names(client):
    """Robustly list collection names across chromadb versions (objects or strings)."""
    names = []
    for c in client.list_collections():
        names.append(getattr(c, "name", str(c)))
    return names


def main() -> None:
    console.rule("[bold cyan]Module 4 - Vector Stores (ChromaDB CRUD)[/]")

    # ---- Setup: embeddings + PERSISTENT client ----------------------------
    try:
        emb = get_embeddings()
    except Exception as exc:
        console.print(Panel(f"[red]Embedding model failed to load:[/]\n{exc}",
                            title="Setup issue", border_style="red"))
        return

    import chromadb
    # PersistentClient stores collections/vectors/metadata under CHROMA_DIR.
    # Under the hood, each collection owns an HNSW index for ANN search.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    console.print(f"\n[bold]Persistent client[/bold] -> {CHROMA_DIR}")

    # Start from a clean slate so the lab is reproducible on every run.
    try:
        client.delete_collection(COLLECTION_NAME)
        console.print(f"  [dim]Deleted stale collection '{COLLECTION_NAME}' for a clean run.[/]")
    except Exception:
        pass  # first run: nothing to delete yet

    # ---- CREATE -----------------------------------------------------------
    console.print("\n[bold]CREATE[/bold] collection")
    # create_collection builds an EMPTY HNSW graph. `hnsw:space` sets the distance metric
    # used both at insert and query time (default is 'l2'; we choose 'cosine' for text).
    col = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_ChromaEF(emb),
        metadata={"hnsw:space": "cosine"},
    )
    console.print(f"  * created '{col.name}'  (count={col.count()})")

    # ---- ADD --------------------------------------------------------------
    console.print("\n[bold]ADD[/bold] documents from data/samples/")
    docs = load_sample_docs()
    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    metas = [d[2] for d in docs]
    # add() embeds each text (via our EF) and inserts vector + metadata into the HNSW graph.
    # Each new node gets edges to its nearest existing nodes (beam width = efConstruction).
    col.add(ids=ids, documents=texts, metadatas=metas)
    console.print(f"  * added {len(ids)} docs  (count={col.count()})")

    # ---- READ (similarity search) -----------------------------------------
    console.print("\n[bold]READ[/bold] - similarity search")
    # Query path: embed the query text -> beam-search the HNSW graph -> return top-k.
    res = col.query(query_texts=["What indexing strategy does ChromaDB use by default?"],
                    n_results=3)
    t = Table(title="Top-3 results (query path: embed -> HNSW beam search)",
              title_style="bold cyan")
    t.add_column("#", justify="right", style="dim")
    t.add_column("dist", justify="right")
    t.add_column("id")
    t.add_column("snippet")
    for rank, (rid, dist, doc) in enumerate(
            zip(res["ids"][0], res["distances"][0], res["documents"][0]), 1):
        t.add_row(str(rank), f"{dist:.3f}", rid, doc[:58].replace("\n", " ") + "...")
    console.print(t)
    # Recall-style sanity check (soft): the vector-db notes doc should rank #1.
    ok = res["ids"][0][0] == "vector_db_notes"
    console.print(f"  [dim]Sanity check: expected 'vector_db_notes' at rank 1 -> "
                  f"{'PASS' if ok else 'WARN (see above)'}[/]")

    # ---- METADATA FILTERING (pre-filter) ----------------------------------
    console.print("\n[bold]METADATA FILTER[/bold] - pre-filtered queries")
    # `where` predicates are applied BEFORE the ANN search (pre-filter), so the HNSW beam
    # only walks candidate nodes that satisfy the predicate.
    r_elec = col.query(query_texts=["best value electronics?"], n_results=3,
                       where={"category": "Electronics"})
    r_apac = col.get(where={"region": "APAC"})
    r_exp = col.get(where={"price_usd": {"$gte": 400}})  # numeric $gte operator
    console.print(f"  Electronics-only top hit : {r_elec['ids'][0][0]}")
    console.print(f"  APAC products in store   : {sorted(r_apac['ids'])}")
    console.print(f"  Products priced >= $400   : {sorted(r_exp['ids'])}")

    # ---- UPDATE (upsert) --------------------------------------------------
    console.print("\n[bold]UPDATE[/bold] - upsert with changed content")
    # upsert() with an EXISTING id replaces the vector + metadata in place. Chroma re-embeds
    # the new text and rewires that node's HNSW edges.
    new_price_doc = "Atlas Standing Desk is now a premium furniture item priced at $649.00."
    col.upsert(ids=["product_2"], documents=[new_price_doc],
               metadatas=[{"source": "products.csv", "type": "product",
                           "category": "Furniture", "region": "NAMER",
                           "price_usd": 649.0, "stock": 35}])
    got = col.get(ids=["product_2"])
    console.print(f"  product_2 now reads     : {got['documents'][0]}")
    console.print(f"  product_2 price_usd now : {got['metadatas'][0]['price_usd']}")

    # ---- DELETE -----------------------------------------------------------
    console.print("\n[bold]DELETE[/bold] - by id and by where-filter")
    before = col.count()
    col.delete(ids=["company_profile"])          # delete a single id
    console.print(f"  deleted 'company_profile'  (count {before} -> {col.count()})")
    col.delete(where={"type": "product"})        # delete ALL products at once
    console.print(f"  deleted all type=product (count {col.count()})")
    remaining = col.get(include=["metadatas"])
    console.print(f"  remaining ids: {sorted(remaining['ids'])}")

    # ---- COLLECTION LIFECYCLE ---------------------------------------------
    console.print("\n[bold]LIFECYCLE[/bold] - list / count / delete collection")
    console.print(f"  collections in store : {collection_names(client)}")
    console.print(f"  '{COLLECTION_NAME}' count = {col.count()}")
    client.delete_collection(COLLECTION_NAME)
    console.print(f"  deleted collection '{COLLECTION_NAME}'")
    console.print(f"  collections now      : {collection_names(client)}")

    console.rule("[bold green]CRUD lab complete [OK][/]")
    console.print(Panel(
        "[dim]Persistence: everything above lived under CHROMA_DIR. Re-open the same "
        "PersistentClient path later and your collections/vectors are still there.\n\n"
        "Note: if you change the embedding MODEL, old vectors are incompatible with the new "
        "space - REBUILD the collection from scratch.[/]",
        title="Under the hood (HNSW)", border_style="blue"))


if __name__ == "__main__":
    main()
