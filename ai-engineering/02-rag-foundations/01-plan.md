# 🎯 Week 02 — RAG Foundations

> Status: ⏳ PENDING (Batch — see `../CHECKPOINT.md`). This is the planning doc; learning + implementation + code land when this week's batch runs.

## Objective
A context-aware RAG pipeline, backed by a local vector store, that accurately answers questions over complex documents.

## What You'll Learn
- Embeddings & Vector Geometry Basics
- Multi-Tenant Isolation Patterns
- Vector Databases & Indexing Strategies
- Document Parsing & Optimal Chunking

## Tools / Stack
- pgvector
- Embeddings

## Weekly Build
**The Knowledge Engine** — A context-aware RAG pipeline, backed by a local vector store, that accurately answers questions over complex documents.

**Outcome:** Build retrieval systems that answer questions factually without hallucinating.

## Deliverables (this week)
- [ ] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [ ] `03-implementation.md` — step-by-step build guide
- [ ] `code/main.py` — framework-based end-to-end build
- [ ] `code/approach_2_*.py` — alternative framework usage
- [ ] `code/approach_3_*.py` — from-scratch implementation
- [ ] `code/*_benchmark.py` — head-to-head comparison (where meaningful)
- [ ] All scripts: Windows UTF-8 guard + graceful degradation + `py_compile` clean

## Prerequisites
- Previous week(s) complete.
- `uv sync` done in `ai-engineering/`; `YOLO_AUTO_API_KEY` set (cloud LLM). No Ollama (8 GB laptop).

## Time Estimate
~4–6 hours hands-on.
