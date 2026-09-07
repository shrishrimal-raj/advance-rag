# 🎯 Week 03 — Enterprise-grade RAG Pipelines

> Status: ✅ BUILT (see `../CHECKPOINT.md`).

## Objective
Push past naive vector search with hybrid strategies, re-ranking, and dynamic query routing.

## What You'll Learn
- Hybrid Search (Semantic + Keyword)
- Cross-Encoder Re-Ranking Strategies
- Query Expansion & Graph RAG Basics

## Tools / Stack
- **rank_bm25** (BM25 keyword) · **sentence-transformers** (dense + cross-encoder rerank) · **fastembed** (alt dense backend) · **Chroma** (store)

## Weekly Build
**The Enterprise Search Engine** — A multi-tenant, re-ranked hybrid search system for enterprise data.

**Outcome:** Master enterprise-grade search techniques used by top AI product teams.

## Approach details
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | Hybrid (BM25 + dense) -> RRF -> optional cross-encoder rerank, multi-tenant | offline (BM25 + cached MiniLM); rerank opt-in |
| `code/approach_2_fastembed_hybrid.py` | Framework-free hybrid; dense backend switchable `st` (default, offline) / `fastembed` | offline by default |
| `code/approach_3_rrf_from_scratch.py` | Reciprocal Rank Fusion from scratch (pure python/numpy) | **fully offline** |
| `code/benchmark_naive_vs_hybrid.py` | naive (dense) vs hybrid (RRF) vs +rerank on gold corpus | offline (rerank opt-in) |

> Note: cross-encoder rerank (`get_reranker`) downloads an ~80 MB model, so it is **opt-in** (`--rerank`) to honor the 8 GB / no-big-download rule. Core hybrid search runs fully offline.

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` · `approach_2_fastembed_hybrid.py` · `approach_3_rrf_from_scratch.py` · `benchmark_naive_vs_hybrid.py`
- [x] All scripts: UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four.
- Run `approach_3` + `benchmark_naive_vs_hybrid` (offline, self-check asserts).
- Run `main.py` (offline hybrid) and `approach_2` (offline, st backend).

## Prerequisites
- Weeks 1–2 complete. `uv sync` done; local MiniLM cached. No Ollama.

## Time Estimate
~4–6 hours hands-on.
