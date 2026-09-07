# 🔨 Module 4 — Implementation Guide

> Build a full ChromaDB CRUD lab in `code/main.py` using the shared **persistent** client
> (`CHROMA_DIR`). No LLM, no API keys — local embeddings + ChromaDB only.

**Run it:**
```bash
uv run python 04-vector-stores/code/main.py
```

---

## Step 0 — Bootstrap + persistent client

```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_embeddings, CHROMA_DIR
import chromadb

emb = get_embeddings()
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
```

- `PersistentClient` writes everything under `CHROMA_DIR` (`./data/chroma_db`) so the lab survives
  restarts. For a reproducible run, **delete the target collection first** (wrapped in try/except).

## Step 1 — Wrap our embeddings for ChromaDB

ChromaDB wants a callable `List[str] -> List[List[float]]`. LangChain's embeddings expose
`embed_documents(list[str])`, so wrap them in a tiny adapter:

```python
class _ChromaEF:
    def __init__(self, lc): self._lc = lc
    def __call__(self, input): return self._lc.embed_documents(list(input))
```

Passing this as `embedding_function` guarantees **add** and **query** use the *same* model.

## Step 2 — CREATE the collection

```python
col = client.create_collection(
    name="course_docs",
    embedding_function=_ChromaEF(emb),
    metadata={"hnsw:space": "cosine"},   # store cosine as the metric
)
```

Creating a collection builds an empty HNSW graph ready to receive vectors.

## Step 3 — ADD documents from `data/samples/`

Load the four sample files into `(id, text, metadata)` tuples:
- `rag_overview.txt` → one article doc.
- `products.csv` → one doc **per row**, with structured metadata (`category`, `region`,
  `price_usd`, `stock`).
- `company_profile.json` → one company doc.
- `vector_db_notes.md` → one notes doc.

Then insert in one batch:
```python
col.add(ids=ids, documents=texts, metadatas=metas)
```
> Under the hood: each text is embedded, and each new vector becomes a node wired into the HNSW
> graph (beam width = `efConstruction`). Check `col.count()` afterwards.

## Step 4 — READ via similarity search

```python
res = col.query(query_texts=["What indexing strategy does ChromaDB use by default?"], n_results=3)
```
Print the top-3 (id, distance, snippet). Add a **soft sanity check**: the `vector_db_notes` doc
should rank #1 (it literally says HNSW is ChromaDB's default). Print PASS/WARN, never assert.

## Step 5 — Metadata filtering (pre-filter)

Show that `where` clauses restrict results *before* the ANN search:
```python
r_elec = col.query(query_texts=["best value electronics?"], n_results=3, where={"category": "Electronics"})
r_apac = col.get(where={"region": "APAC"})
r_exp  = col.get(where={"price_usd": {"$gte": 400}})   # numeric operator
```
Print the resulting ids — only matching docs should appear.

## Step 6 — UPDATE via upsert

Replace an existing doc's content + metadata in place:
```python
col.upsert(ids=["product_2"], documents=[new_text], metadatas=[{... "price_usd": 649.0 ...}])
got = col.get(ids=["product_2"])
```
> Upsert re-embeds the new text and rewires that node's HNSW edges. Verify the change with `get`.

## Step 7 — DELETE by id and by where-filter

```python
col.delete(ids=["company_profile"])      # remove one id
col.delete(where={"type": "product"})    # remove ALL products at once
```
Print `col.count()` before/after and the remaining ids.

## Step 8 — Collection lifecycle

```python
names = [c.name for c in client.list_collections()]
print(col.count())
client.delete_collection("course_docs")
```
Show list → count → delete, and confirm the collection is gone. Finish with a panel explaining
persistence + the "rebuild on model change" rule.

---

## ✅ Verification checklist

- [ ] Script runs with no errors; every CRUD verb executes.
- [ ] `count()` grows after ADD and shrinks after each DELETE.
- [ ] The similarity-search sanity check prints PASS (or a visible WARN).
- [ ] Filtered queries return only matching ids (Electronics / APAC / price ≥ 400).
- [ ] After upsert, `product_2` shows the new price.
- [ ] Collection appears in `list_collections()`, then disappears after `delete_collection`.

## 🧪 Experiments to try

1. Re-run the script twice — confirm the stale-collection cleanup makes it reproducible.
2. Change `hnsw:space` to `"l2"` and compare distances to the cosine run.
3. Add a very selective filter (e.g. `where={"region":"EMEA","category":"Furniture"}`) and watch
   the candidate set shrink.
4. Inspect `./data/chroma_db` on disk to see the persisted index files.
