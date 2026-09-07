# 🎯 Module 3 — Embeddings & Vector Representations

> **Goal:** Understand *how* text becomes a dense vector, *why* similar meanings cluster in
> vector space, and *how* to measure similarity — then prove it hands-on with NumPy.

---

## 🧭 Objectives

By the end of this module you will be able to:

1. Explain the full pipeline that turns raw text into a dense vector
   (tokenization → transformer encoder → pooling → normalization).
2. Describe **embedding-space geometry** and why semantically similar sentences cluster together.
3. Write down and correctly apply the three core distance metrics:
   **cosine similarity**, **L2 / Euclidean distance**, and **inner product**.
4. Explain the **normalization subtlety**: why cosine ≈ inner product once vectors are unit-length.
5. Compare **proprietary** (OpenAI, Cohere) vs **open-source** (MiniLM, BGE, E5, GTE) models on
   quality / cost / latency / dimensions.
6. Make a defensible **model choice** for a given budget, language, and domain.

## ✅ Prerequisites

- **Module 1** (RAG Fundamentals): you know the `Knowledge Base → Retriever → Generator` flow.
- **Module 2** (Document Processing & Chunking): you know what a "document/chunk" is.
- Basic linear-algebra intuition: vectors, dot product, length/norm.
- Environment ready: `uv sync` completed; you can download the local embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB) on first run.
- **No API key required** — this module uses only local embeddings + NumPy (no LLM).

## 📦 Deliverables Checklist

- [ ] Read `02-learning.md` end-to-end (theory + both Mermaid diagrams).
- [ ] Run `code/main.py` successfully (`uv run python 03-embeddings-vector-representations/code/main.py`).
- [ ] Interpret the three pairwise matrices (cosine / L2 / inner product) for the 8 sample sentences.
- [ ] Confirm the lab shows cosine ≈ inner product after L2-normalization (max diff ≈ 0).
- [ ] Read the nearest-neighbour result and explain *why* the top hit is the RAG sentence.
- [ ] Note your model-choice reasoning in `notes.md`.

## ⏱️ Time Estimate

| Phase | Activity | Time |
|-------|----------|------|
| 1 | Read theory + diagrams | ~45 min |
| 2 | Run & interpret the lab | ~45 min |
| 3 | Experiments (change sentences, try a bigger model) | ~30 min |
| **Total** | | **~2 hours** |

## 🏁 Success Criteria

You have mastered this module when you can:

- **Explain** the tokenization → model → vector pipeline without looking at notes.
- **State** when to use cosine vs L2 vs inner product, and *why*.
- **Run** the lab and see all 5 steps print cleanly.
- **Justify** choosing 384-dim MiniLM over a 1024/1536-dim model (or vice-versa) for a real project.
- **Predict** which of two sentences is closer to a query by reasoning about their embeddings.
