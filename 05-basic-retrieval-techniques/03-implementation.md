# 🔨 Module 5 — Implementation Guide

You will build **one shared corpus** and run **three identical queries** through **five
retrievers**, printing a side-by-side comparison. Everything lives in `code/main.py`.

> Run from the project root:
> ```bash
> uv run python 05-basic-retrieval-techniques/code/main.py
> ```

---

## Step 1 — Bootstrap & imports

At the top of `code/main.py` we make `shared.config` importable from anywhere and pull in the
pieces we need:

```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_embeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rank_bm25 import RankBM25
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
```

- `parents[2]` walks `code/ → module/ → project-root/` so `shared.config` resolves.
- We use **`langchain_community.vectorstores.Chroma`** (already a dependency) — no extra install.
- **No LLM import** — this module is embeddings-only.

## Step 2 — Build the shared corpus

Three small functions do load → chunk → embed:

1. **`load_raw_documents()`** reads every file in `data/samples/` into a `Document`, tagging
   `metadata = {"source": filename, "format": ext}`.
2. **`build_chunks(raw_docs)`** runs a `RecursiveCharacterTextSplitter`
   (`chunk_size=300, chunk_overlap=40`) and stamps each chunk with a stable
   `metadata["id"] = "<source>::chunk<i>"`. Stable IDs matter — they're the join key for RRF.
3. **`build_vectorstore(chunks)`** calls
   `Chroma.from_documents(chunks, get_embeddings(), collection_name="m5_corpus",
   collection_metadata={"hnsw:space": "cosine"})`.
   - **In-memory** (no `persist_directory`) → nothing hits disk.
   - **`hnsw:space = "cosine"`** forces cosine distance, so `1 − distance` is a clean similarity.

## Step 3 — Build the sparse (BM25) index

```python
def tokenize(text): return re.findall(r"[a-z0-9]+", text.lower())
bm25 = RankBM25([tokenize(c.page_content) for c in chunks])
```

We keep the **same `chunks` list** as the ground truth for both the vector store and BM25, so the
two branches are searching the *identical* documents — a fair comparison.

## Step 4 — Implement Reciprocal Rank Fusion

```python
def rrf_fuse(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
```

Each input is an **ordered list of doc IDs** (best first). A doc appearing near the top of several
lists accumulates the highest score. `k=60` is the canonical constant.

## Step 5 — The five retriever functions

Each returns a list of `(rank, score, source, snippet)` rows for pretty printing:

| Function | What it calls | Notes |
|----------|---------------|-------|
| `r_similarity` | `vs.similarity_search_with_score(q, k=3)` | flips distance → similarity |
| `r_threshold` | `vs.similarity_search_with_score(q, k=n)` then keeps `sim ≥ 0.45` | scans **all** `n` so off-topic → empty |
| `r_mmr` | `vs.max_marginal_relevance_search(q, k=3, fetch_k=20, lambda_mult=0.5)` | no scores (diversity) |
| `r_bm25` | `bm25.get_scores(tokenize(q))` → top-3 indices | raw BM25 scores |
| `r_hybrid` | RRF over dense-top-10 + BM25-top-10 | rank-based fusion |
| `r_ensemble` | min-max normalize both branches, `0.5·dense + 0.5·sparse` | score-based fusion |

**Threshold detail:** we deliberately fetch the *whole* collection (`k=n`) before filtering, so a
genuinely off-topic query has a chance to return **zero** rows — that's the teaching moment.

## Step 6 — Drive the comparison

In `main()` we define three queries and loop:

```python
queries = [
    "How does RAG reduce hallucination?",          # on-topic (rag_overview.txt)
    "What are HNSW parameters like M and efSearch?",# on-topic (vector_db_notes.md)
    "How do I bake chocolate chip cookies?",       # OFF-topic -> trips the gate
]
```

For each query we print all five (plus the ensemble variant) as Rich tables. Watch:

- Query 3 returns **empty** under the threshold retriever but still returns rows elsewhere —
  proof of why the gate exists.
- Query 2 shows **BM25** latching onto `HNSW`/`efSearch` hard.
- MMR tends to spread results across **different sources** than plain top-k.

## Step 7 — Graceful failure

The `__main__` block wraps `main()` in `try/except` and prints a friendly panel if the embedding
model can't be downloaded or a dependency is missing.

---

## 🧪 Suggested experiments (do these!)

1. **λ sweep:** change `lambda_mult` in `r_mmr` to `0.0`, `0.5`, `1.0`; watch diversity collapse/grow.
2. **Threshold sweep:** change `min_score` to `0.30`, `0.45`, `0.60`; see how many queries go empty.
3. **RRF constant:** change `k_rrf` from `60` to `10`; notice the top-of-list dominance increase.
4. **Ensemble weights:** bias to `w_dense=0.8, w_sparse=0.2` and compare to RRF ordering.

Record your observations in `notes.md`.
