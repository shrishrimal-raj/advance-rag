# 🎯 Week 02 — RAG Foundations

> Status: ✅ BUILT (see `../CHECKPOINT.md`). Planning doc; learning + implementation + code below.

## Objective
A context-aware RAG pipeline, backed by a local vector store, that accurately answers questions over complex documents.

## What You'll Learn
- Embeddings & Vector Geometry Basics
- Multi-Tenant Isolation Patterns
- Vector Databases & Indexing Strategies
- Document Parsing & Optimal Chunking

## Tools / Stack
- pgvector (concept) · **Chroma** (local stand-in, installed) · sentence-transformers MiniLM (local embeddings)

## Weekly Build
**The Knowledge Engine** — A context-aware RAG pipeline, backed by a local vector store, that accurately answers questions over complex documents.

**Outcome:** Build retrieval systems that answer questions factually without hallucinating.

## Approach details (multi-approach, per course convention)
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | Framework: LangChain + Chroma + local MiniLM + cloud LLM, per-tenant collections | LLM (1 call) + cached embeddings |
| `code/approach_2_raw_vector_db.py` | Raw Chroma client + sentence-transformers directly (no LangChain) | cached embeddings, **no LLM** |
| `code/approach_3_cosine_from_scratch.py` | From-scratch TF-IDF vectorizer + cosine top-k in pure numpy | **fully offline** |
| `code/chunking_benchmark.py` | Head-to-head chunking strategies scored by a from-scratch retriever | **fully offline** |

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` — framework-based end-to-end build
- [x] `code/approach_2_raw_vector_db.py` — raw vector DB API
- [x] `code/approach_3_cosine_from_scratch.py` — from-scratch cosine retrieval
- [x] `code/chunking_benchmark.py` — chunking-strategy comparison
- [x] All scripts: Windows UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four scripts.
- Run `approach_3` + `chunking_benchmark` (offline, must pass incl. self-check asserts).
- Run `approach_2` (cached MiniLM, no LLM).
- Run `main.py --selftest` (one cloud LLM call; degrades gracefully if key absent).

## Prerequisites
- Week 1 complete. `uv sync` done in `ai-engineering/`; `YOLO_AUTO_API_KEY` set (cloud LLM). No Ollama (8 GB laptop).

## Time Estimate
~4–6 hours hands-on.
