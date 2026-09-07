# PLANNING — Enterprise Search Service

## Problem
Single-method search misses things: pure keyword search fails on paraphrase; pure
vector search can miss exact identifiers. Enterprise search needs both, fused well.

## Goals
1. Implement BM25 sparse scoring from scratch.
2. Implement a dense cosine path (deterministic proxy, swappable for real embeddings).
3. Fuse with Reciprocal Rank Fusion.
4. Optional cross-encoder-style rerank of the top-k.
5. Fully testable offline.

## Non-goals
- Distributed indexing / sharding (v2).
- Real-time ingestion pipeline (v2).
- Query understanding / rewriting (v2).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | BM25 from scratch | ✅ |
| M2 | Dense cosine path | ✅ |
| M3 | RRF fusion | ✅ |
| M4 | Rerank stage | ✅ |
| M5 | Offline tests + API + Docker + CI | ✅ |

## Risks & mitigations
- Dense proxy weaker than real embeddings → documented swap-in point; RRF/rerank unchanged.
- Score-scale mismatch between methods → RRF is rank-based, immune to scale.
