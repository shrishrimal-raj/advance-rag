# PLANNING — The Knowledge Engine

## Problem
Naive keyword search misses semantic matches; naive vector search misses exact terms
(error codes, product names). Production RAG needs both, plus scoping by metadata and
traceable citations.

## Goals
1. Hybrid retrieval: BM25 (sparse) + dense cosine, fused by RRF.
2. Metadata filtering applied before ranking (tenant/topic scoping).
3. Citation tracking: every hit carries id + source + position.
4. LLM-ready context builder.
5. Fully testable offline (no network, no pgvector).

## Non-goals
- Real embedding model calls (stand-in vectors; swap-in point documented).
- Re-ranking model (RRF is the fusion; add a cross-encoder later).
- Multi-tenant auth (metadata filter is the scoping primitive).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | Chunk model + tokenizer + cosine | ✅ |
| M2 | Hybrid BM25+dense + RRF fusion | ✅ |
| M3 | Metadata filtering | ✅ |
| M4 | Citation tracking + context builder | ✅ |
| M5 | FastAPI + Docker/compose(pgvector) + CI | ✅ |

## Risks & mitigations
- rank_bm25 missing → token-overlap fallback keeps tests green.
- Vector quality → deterministic stand-ins for tests; real embeddings in prod.
