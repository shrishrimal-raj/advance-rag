# 🎯 Module 5 — Basic Retrieval Techniques (Retrievers)

> **Goal:** Move from "I have a vector database" to "I know *how* to pull the right
> chunks out of it." This module builds one shared corpus and runs the **same** queries
> through **five** fundamental retrievers so you can *see* how each one behaves differently.

---

## 🎯 Objectives

By the end of this module you will be able to:

1. Explain what a **retriever** is and why it is a first-class abstraction (not just a DB call).
2. Use **Similarity Search** (`top-k`) and read its **scores**.
3. Use a **Similarity Score Threshold** retriever and explain *when returning nothing* is a feature (anti-hallucination).
4. Apply **Maximal Marginal Relevance (MMR)** and tune `lambda_mult` to trade relevance vs. diversity.
5. Implement **BM25 sparse retrieval** and explain why keywords still win for names, IDs, and error codes.
6. Build **Hybrid Search** (dense + sparse) fused with **Reciprocal Rank Fusion (RRF, k=60)**.
7. Assemble an **Ensemble** of retrievers with weights and compare it to RRF.

## 🧩 Prerequisites

- ✅ Module 4 (Vector Stores) — you can create a Chroma collection and query it.
- ✅ Module 3 (Embeddings) — you understand cosine similarity and distance metrics.
- ✅ `uv sync` completed; local embedding model (`all-MiniLM-L6-v2`) downloadable.
- ⚠️ **No LLM / no API key needed** for this module — only local embeddings.

## 📦 Deliverables Checklist

- [ ] Read `02-learning.md` (theory + 2 architecture diagrams).
- [ ] Follow `03-implementation.md` step by step.
- [ ] Run `code/main.py` and confirm **all five** retrievers print results for **three** queries.
- [ ] Observe the **off-topic query** returning **empty** under the threshold retriever.
- [ ] Tune `lambda_mult` (try `0.0`, `0.5`, `1.0`) and note how MMR output changes.
- [ ] Write your findings in `notes.md`.

## ⏱️ Time Estimate

| Activity | Time |
|----------|------|
| Reading theory + diagrams | 30 min |
| Running & reading the comparison output | 20 min |
| Experiments (λ tuning, threshold sweep) | 30 min |
| Notes | 10 min |
| **Total** | **~1.5 hours** |

## ✅ Success Criteria

You're done when you can answer, *without looking*:

- Why does a threshold retriever sometimes return **zero** results, and why is that good?
- What does `lambda_mult = 1.0` vs `0.0` do to MMR?
- Why does BM25 find the string `HNSW` / `efSearch` reliably while a dense model might not?
- What is the RRF formula and why is `k = 60`?
- When would you pick hybrid over pure dense?
