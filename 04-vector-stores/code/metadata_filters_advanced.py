"""
Module 4 - Vector Stores: Metadata filtering deep-dive (raw ChromaDB API).

Run from the project root:
    uv run python 04-vector-stores/code/metadata_filters_advanced.py

Why this exists:
    In production RAG you almost never retrieve on vectors ALONE. You combine
    semantic similarity with SCALAR predicates: "only docs from the pricing
    guide", "only pages after 2024-01-01", "only internal-access docs". This
    script builds a small collection with REALISTIC metadata and exercises
    every ChromaDB filter operator, printing exactly which docs match.

Operators covered (all via raw chromadb `where=`):
    equality        {"source": "pricing_guide.pdf"}
    $in             {"primary_tag": {"$in": ["pricing", "api"]}}
    negation        {"access_level": {"$ne": "public"}}   ($not was removed in chromadb 1.5;
                    {"primary_tag": {"$nin": [...]}}        use $ne / $nin instead)
    $gte / $lte     {"page": {"$gte": 5}}          (numeric ranges)
                    {"date_key": {"$gte": 20240101}} (int yyyymmdd - chromadb 1.5
                    rejects STRING range ops, so dates are stored as ints too)
    $and / $or      compound predicates

Key production facts printed along the way:
    - Filters are PRE-filtered: Chroma prunes candidates before the HNSW walk,
      so a selective filter can actually make queries FASTER (and more correct).
    - Negation is $ne / $nin in chromadb 1.5.x (the old $not operator is gone).
    - Range ops ($gte/$lte) only accept NUMBERS in chromadb 1.5 - store dates as
      int yyyymmdd (or epoch seconds) if you want to filter by date.
    - Filter on SCALAR fields: in chromadb 1.5, $in on list-valued metadata
      (e.g. tags=[...]) silently matches nothing - keep one scalar 'primary_tag'.

NO LLM / NO API keys - local MiniLM embeddings + ChromaDB only.
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
TEMP_DIR = PROJECT_ROOT / "data" / "chroma_meta_filters"   # temp persist dir under data/
COLLECTION_NAME = "meta_filter_lab"


class _ChromaEF:
    """Adapter: expose a LangChain embeddings object as a ChromaDB EmbeddingFunction."""
    def __init__(self, lc):
        self._lc = lc

    def __call__(self, input):
        return self._lc.embed_documents(list(input))

    def embed_query(self, input):
        items = list(input)
        if len(items) == 1:
            return [self._lc.embed_query(items[0])]
        return self._lc.embed_documents(items)


# Realistic doc set: (id, text, metadata). primary_tag is SCALAR on purpose:
# chromadb 1.5 cannot $in-filter list-valued metadata reliably.
DOCS = [
    ("pg_1", "The enterprise plan costs $99 per seat billed annually.",
     {"source": "pricing_guide.pdf", "page": 1, "date": "2024-03-15", "date_key": 20240315,
      "tags": ["pricing", "plans"], "primary_tag": "pricing", "access_level": "public"}),
    ("pg_2", "Volume discounts start at 50 seats and require sales contact.",
     {"source": "pricing_guide.pdf", "page": 2, "date": "2024-03-15", "date_key": 20240315,
      "tags": ["pricing", "discounts"], "primary_tag": "pricing", "access_level": "public"}),
    ("api_1", "The /v1/search endpoint accepts a query string and top_k parameter.",
     {"source": "api_reference.md", "page": 10, "date": "2024-06-01", "date_key": 20240601,
      "tags": ["api", "search"], "primary_tag": "api", "access_level": "public"}),
    ("api_2", "Rate limits are 60 requests per minute on the free tier.",
     {"source": "api_reference.md", "page": 12, "date": "2024-06-01", "date_key": 20240601,
      "tags": ["api", "limits"], "primary_tag": "api", "access_level": "public"}),
    ("int_1", "Internal incident postmortem: index rebuild caused 40s latency spike.",
     {"source": "internal_notes.md", "page": 3, "date": "2023-11-20", "date_key": 20231120,
      "tags": ["incident", "performance"], "primary_tag": "incident", "access_level": "internal"}),
    ("int_2", "Roadmap draft: hybrid retrieval ships in Q3.",
     {"source": "internal_notes.md", "page": 7, "date": "2024-08-10", "date_key": 20240810,
      "tags": ["roadmap"], "primary_tag": "roadmap", "access_level": "internal"}),
    ("hr_1", "Employee handbook section on remote work policy.",
     {"source": "handbook.pdf", "page": 22, "date": "2023-05-01", "date_key": 20230501,
      "tags": ["policy"], "primary_tag": "policy", "access_level": "internal"}),
    ("pg_3", "Legacy 2022 price list, superseded by the current guide.",
     {"source": "pricing_guide.pdf", "page": 5, "date": "2022-12-01", "date_key": 20221201,
      "tags": ["pricing", "legacy"], "primary_tag": "pricing", "access_level": "public"}),
]


def show(label: str, where: dict, col, query_text: str | None = None) -> None:
    """Run a filtered read and print which docs matched."""
    if query_text is not None:
        res = col.query(query_texts=[query_text], n_results=len(DOCS), where=where)
        ids = res["ids"][0]
    else:
        res = col.get(where=where)
        ids = res["ids"]
    t = Table(title=f"{label}  ->  {len(ids)} match(es)", title_style="bold cyan")
    t.add_column("id")
    t.add_column("source")
    t.add_column("page", justify="right")
    t.add_column("date")
    t.add_column("access")
    for rid in ids:
        m = {d[0]: d[2] for d in DOCS}[rid]
        t.add_row(rid, m["source"], str(m["page"]), m["date"], m["access_level"])
    console.print(t)
    console.print(f"  [dim]where={where}[/]\n")


def main() -> None:
    console.rule("[bold cyan]Module 4 - Metadata filters deep-dive (raw chromadb)[/]")

    try:
        emb = get_embeddings()
    except Exception as exc:
        console.print(Panel(
            f"[red]Embedding model failed to load:[/]\n{exc}\n\n"
            "[bold]Hint:[/bold] needs the local MiniLM model (caches after first download).",
            title="Setup issue", border_style="red"))
        sys.exit(0)

    import chromadb

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    client = chromadb.PersistentClient(path=str(TEMP_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(name=COLLECTION_NAME,
                                   embedding_function=_ChromaEF(emb),
                                   metadata={"hnsw:space": "cosine"})
    col.add(ids=[d[0] for d in DOCS], documents=[d[1] for d in DOCS],
            metadatas=[d[2] for d in DOCS])
    console.print(f"\nLoaded {col.count()} docs into '{COLLECTION_NAME}' "
                  f"(temp dir: {TEMP_DIR})\n")

    # ---- 1. Equality ---------------------------------------------------------
    show("EQUALITY: all docs from the pricing guide",
         {"source": "pricing_guide.pdf"}, col)

    # ---- 2. $in ----------------------------------------------------------------
    show("$IN: sources in a list",
         {"source": {"$in": ["api_reference.md", "handbook.pdf"]}}, col)
    show("$IN on scalar tag field: pricing OR api docs",
         {"primary_tag": {"$in": ["pricing", "api"]}}, col)

    # ---- 3. Negation ($ne / $nin - chromadb 1.5 removed the old $not) --------------
    show("NEGATION $ne: everything that is NOT public",
         {"access_level": {"$ne": "public"}}, col)
    show("NEGATION $nin: nothing tagged pricing or api",
         {"primary_tag": {"$nin": ["pricing", "api"]}}, col)

    # ---- 4. Numeric ranges ($gte / $lte) --------------------------------------------
    show("$GTE: pages numbered 5 or higher",
         {"page": {"$gte": 5}}, col)
    show("$LTE: pages numbered 12 or lower",
         {"page": {"$lte": 12}}, col)
    show("$gte + $lte combined: pages 5..12 (two ops on one field need $and in chromadb 1.5)",
         {"$and": [{"page": {"$gte": 5}}, {"page": {"$lte": 12}}]}, col)

    # ---- 5. Date ranges (stored as int yyyymmdd - chromadb 1.5 has no string ranges) --
    show("DATE $gte: docs dated 2024-01-01 or later (date_key >= 20240101)",
         {"date_key": {"$gte": 20240101}}, col)
    show("DATE range: 2024-01-01 .. 2024-12-31",
         {"$and": [{"date_key": {"$gte": 20240101}}, {"date_key": {"$lte": 20241231}}]}, col)

    # ---- 6. Compound $and / $or ----------------------------------------------------------
    show("$AND: public AND page >= 10",
         {"$and": [{"access_level": "public"}, {"page": {"$gte": 10}}]}, col)
    show("$OR: tagged 'pricing' OR source is handbook",
         {"$or": [{"primary_tag": {"$eq": "pricing"}}, {"source": "handbook.pdf"}]}, col)
    show("$AND + $OR nested: internal AND (incident OR roadmap)",
         {"$and": [{"access_level": "internal"},
                   {"$or": [{"primary_tag": {"$eq": "incident"}},
                            {"primary_tag": {"$eq": "roadmap"}}]}]}, col)

    # ---- 7. Filters COMBINED WITH similarity search ----------------------------------------
    console.print("[bold]FILTERS + VECTORS together[/bold] (the real RAG pattern)\n")
    show("Query 'how much does it cost?' restricted to public docs",
         {"access_level": "public"}, col, query_text="How much does the plan cost?")
    show("Same query restricted to the pricing guide only",
         {"source": "pricing_guide.pdf"}, col, query_text="How much does the plan cost?")

    client.delete_collection(COLLECTION_NAME)
    client.close()   # release SQLite handles so the temp dir can be removed on Windows
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except Exception:
        pass

    console.rule("[bold green]Metadata filter lab complete [OK][/]")
    console.print(Panel(
        "[dim]Production notes:\n"
        "- Pre-filtering means selective predicates rarely hurt recall; they prune the graph.\n"
        "- Keep filterable fields TYPED consistently; range ops need NUMBERS (dates as yyyymmdd).\n"
        "- Filter on SCALAR fields: $in on list-valued metadata is unreliable in chromadb 1.5.\n"
        "- For multi-tenant SaaS, ALWAYS include the tenant id in every `where` clause.[/]",
        title="Under the hood", border_style="blue"))


if __name__ == "__main__":
    main()
