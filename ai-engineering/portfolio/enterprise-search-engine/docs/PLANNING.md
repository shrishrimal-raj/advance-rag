# PLANNING — The Enterprise Search Engine

## Problem
Enterprise search must serve many customers from one index while guaranteeing that no
tenant ever sees another tenant's data. It also needs hybrid recall (exact terms +
semantics) and traceable citations.

## Goals
1. Hard multi-tenant isolation (namespace per tenant).
2. Hybrid BM25 + dense cosine retrieval with RRF fusion.
3. Metadata filtering within a tenant.
4. Citation tracking on every hit.
5. Fully offline-testable (no Pinecone, no secrets).

## Non-goals
- Real Pinecone client calls (namespace emulation; swap-in point documented).
- AuthN/AuthZ (the `X-Tenant-ID` header is the trust boundary; put a gateway in front).
- Re-ranking model (RRF only).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | Doc model + tokenizer + cosine | ✅ |
| M2 | Multi-tenant namespaces + upsert | ✅ |
| M3 | Hybrid BM25+dense + RRF | ✅ |
| M4 | Metadata filter + citations | ✅ |
| M5 | FastAPI (X-Tenant-ID) + Docker + CI | ✅ |

## Risks & mitigations
- Cross-tenant leak → search() only reads the requested namespace; test asserts no leak.
- rank_bm25 missing → token-overlap fallback keeps tests green.
