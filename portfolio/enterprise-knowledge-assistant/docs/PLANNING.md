# PLANNING — Enterprise Knowledge Assistant

## Problem statement

Enterprises keep institutional knowledge in markdown documents (runbooks, policy
notes, product docs) that employees cannot search effectively. A naive
"stuff everything into a prompt" approach does not scale, and a single-vector
RAG pipeline answers with low-precision passages, leaks PII into LLM prompts,
and pays full LLM cost for near-duplicate questions.

This project builds a **production-grade RAG service** that answers questions
grounded in a markdown knowledge base with citations, while staying safe and
cost-efficient:

- **Hybrid retrieval** (BM25 sparse + dense vectors, fused with Reciprocal Rank
  Fusion) so exact-term and semantic queries both rank well.
- **Guardrails** (PII redaction + prompt-injection blocking) run *before* any
  LLM call, so personal data never reaches the model and blocked requests cost
  nothing.
- **Semantic response cache** absorbs near-duplicate questions without paying
  LLM cost.
- **Graceful degradation**: with no API key configured, `/ask` returns top
  retrieved passages instead of failing; with an empty knowledge base it
  returns `503` with an actionable hint.

## Goals

| # | Goal | How it is met in code |
|---|------|-----------------------|
| G1 | Grounded, cited answers | `app/main.py` builds numbered `[n]` context and a system prompt requiring `[n]` citations; response includes `citations` (source paths) |
| G2 | High-precision retrieval | `HybridRetriever.search()` in `app/retrieval.py`: BM25 + dense → `_rrf_fuse()` → optional cross-encoder rerank |
| G3 | Safe public endpoint | `sanitize_question()` in `app/guardrails.py`: injection patterns → HTTP 400; PII redacted to typed placeholders before retrieval/generation |
| G4 | Cost control | `SemanticCache` in `app/cache.py` (cosine-similarity hit at `CACHE_THRESHOLD`, default 0.92) |
| G5 | Offline-testable core | `tests/test_retrieval.py` covers chunking, RRF math, and end-to-end retrieval with no LLM and no network |
| G6 | Deployable artifact | `Dockerfile`, `docker-compose.yml`, systemd unit in `docs/DEPLOYMENT.md`, GitHub Actions CI |

## Non-goals

- No multi-tenant auth / per-user permissions (single trusted deployment).
- No document upload API — ingestion is batch (`HybridRetriever.ingest_files`)
  over `.md` files, not a runtime write path.
- No streaming responses; `/ask` is a single synchronous completion.
- No fine-tuning or custom embedding training — local MiniLM embeddings only.
- No web UI; the product surface is the JSON API (`GET /health`, `POST /ask`).

## Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| R1 | `POST /ask` accepts `{question, k}` (question 3–2000 chars, k 1–10) and returns answer, citations, cached flag | Must | Done (`AskRequest` in `app/main.py`) |
| R2 | Answers cite numbered context passages; insufficient context must be stated explicitly | Must | Done (system prompt in `app/main.py`) |
| R3 | Hybrid BM25+dense retrieval with RRF fusion, scores normalized to (0, 1] | Must | Done (`_rrf_fuse` in `app/retrieval.py`) |
| R4 | Persistent vector store surviving restarts | Must | Done (ChromaDB `PersistentClient`, `data/chroma_db` by default) |
| R5 | PII (email, phone, SSN, credit card, IP) redacted from questions before LLM use | Must | Done (`redact_pii` in `app/guardrails.py`) |
| R6 | Prompt-injection phrasings rejected with 400 before retrieval | Must | Done (`detect_injection` + 400 in `app/main.py`) |
| R7 | Semantic cache with configurable threshold; near-duplicates replace instead of duplicating | Should | Done (`SemanticCache.put` in `app/cache.py`) |
| R8 | Optional cross-encoder rerank, off by default (~80MB model download) | Should | Done (`RERANK_ENABLED`, `_rerank` in `app/retrieval.py`) |
| R9 | Degraded mode without an LLM key: return top passages with `degraded: true` | Should | Done (`/ask` in `app/main.py`) |
| R10 | `GET /health` reports provider name and doc count | Should | Done |
| R11 | All tunables via environment / `.env`; no hard-coded secrets | Must | Done (`Settings` dataclass in `app/config.py`) |
| R12 | CI runs the offline test suite on every push | Should | Done (`.github/workflows/ci.yml`) |
| R13 | Multi-language document support (non-English corpora) | Could | Not started |
| R14 | Query rewriting / HyDE expansion stage | Could | Not started |

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| M1 — Retrieval core | Chunking, Chroma persistence, BM25+dense+RRF, offline tests | ✅ Done |
| M2 — Service layer | FastAPI app, guardrails, semantic cache, LLM client, degraded mode | ✅ Done |
| M3 — Production hardening | Dockerfile, docker-compose, CI, deployment doc, planning/design docs | ✅ Done |
| M4 — Quality tuning (future) | Rerank A/B vs RRF-only, cache threshold calibration, query rewriting | ⬜ Planned |
| M5 — Scale-out (future) | External Chroma server, horizontal replicas behind LB, observability dashboards | ⬜ Planned |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stale KB gives confident-wrong answers | Medium | High | Grounded system prompt ("say so explicitly" when context insufficient); citations let users verify; nightly `chroma_db/` backup + re-ingest process |
| Prompt injection via retrieved documents | Medium | High | Injection filter on the question channel; system prompt restricts answers to numbered passages only |
| Semantic cache serves stale answers after KB updates | Medium | Medium | `CACHE_THRESHOLD` is the safety dial (lower = fresher); cache is in-memory and cleared on restart |
| Embedding model download (~90MB) fails on air-gapped hosts | Low | Medium | Model is disk-cached after first run; pin `LOCAL_EMBEDDING_MODEL` and pre-bake into the image if needed |
| BM25 index rebuilt in memory doesn't scale past ~100k chunks | Low | Medium | Documented limit in `app/retrieval.py`; M5 moves to external store |
| LLM provider outage / rate limits | Medium | Medium | 180s timeout + `LLMError` surfaced as 5xx; provider priority OPENAI → Yolo-Auto allows failover by key rotation |
| Secrets committed accidentally | Low | Critical | `.env` gitignored; only `.env.example` with placeholders tracked; CI never reads real keys |

## Success metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test suite green in CI, fully offline | 100% of runs | `uv run pytest -v` in `.github/workflows/ci.yml` |
| Retrieval precision on sample queries | Top-k contains the expected source | `test_search_hits_expected_doc` pattern; extend to a golden set in M4 |
| Cache hit rate on repetitive question traffic | ≥ 30% | `SemanticCache.hit_rate` property (expose via metrics in M5) |
| Guardrail false-positive rate on legitimate questions | < 1% | Sample audit of blocked 400s in logs |
| p95 `/ask` latency (cache miss, LLM on) | < 10s | Load test in M4 |
| Availability of `/health` | 99.5% monthly | Uptime check against deployed URL |
