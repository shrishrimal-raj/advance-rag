# 🎯 Module 4 — Vector Stores

> **Goal:** Understand *why* we need a dedicated vector database, how ANN indexes (Flat / IVF /
> HNSW) trade speed vs accuracy, and how to run full CRUD + metadata filtering in **ChromaDB**.

---

## 🧭 Objectives

By the end of this module you will be able to:

1. Explain what a vector store is and why a dedicated DB beats brute-force search at scale.
2. Describe the three main ANN indexing strategies — **Flat**, **IVF**, **HNSW** — and their key
   parameters (`nlist`/`nprobe`, `M`/`efConstruction`/`efSearch`).
3. Choose a **distance metric** and understand **pre-filter vs post-filter** metadata search.
4. Perform full **CRUD** (Create / Read / Update / Delete) in ChromaDB with a **persistent client**.
5. Explain what happens **under the hood** (HNSW graph) during add and query.
6. Pick the right production vector DB (Qdrant / pgvector / Milvus / Weaviate / Pinecone) for a given need.

## ✅ Prerequisites

- **Module 3** (Embeddings): you can produce and compare dense vectors.
- Basic understanding of graphs (nodes + edges) helps for HNSW.
- Environment ready: `uv sync` completed; local embedding model available.
- **No API key required** — local embeddings + local ChromaDB only.

## 📦 Deliverables Checklist

- [ ] Read `02-learning.md` (indexing strategies + both Mermaid diagrams + alternatives table).
- [ ] Run `code/main.py` successfully (`uv run python 04-vector-stores/code/main.py`).
- [ ] Observe CREATE → ADD → READ → UPDATE → DELETE all working against `data/samples/`.
- [ ] See a metadata-filtered query return only matching docs.
- [ ] Confirm persistence: data lives under `CHROMA_DIR` (`./data/chroma_db`).
- [ ] Note your vector-DB choice reasoning in `notes.md`.

## ⏱️ Time Estimate

| Phase | Activity | Time |
|-------|----------|------|
| 1 | Read theory + diagrams | ~50 min |
| 2 | Run & trace the CRUD lab | ~40 min |
| 3 | Experiments (filters, re-open persistence, tune params) | ~30 min |
| **Total** | | **~2 hours** |

## 🏁 Success Criteria

You have mastered this module when you can:

- **Explain** why brute force is O(n) per query and how HNSW gets sub-millisecond lookups.
- **State** what `M`, `efConstruction`, and `efSearch` control and how to tune them.
- **Run** the lab and watch every CRUD operation succeed with correct counts.
- **Write** a pre-filtered query and explain why filtering before search is faster.
- **Justify** choosing pgvector vs Qdrant vs Pinecone for a real project.
