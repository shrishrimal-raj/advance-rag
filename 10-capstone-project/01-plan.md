# 🎯 Module 10 — Capstone: "Papeer" Research Assistant

> **Goal:** Build **Papeer**, a production-ready research-assistant RAG system that answers
> questions over a document corpus using hybrid retrieval, cross-encoder reranking, an
> agentic (LangGraph) workflow, RAGAS evaluation, and a FastAPI service layer.

This is the capstone. Everything from Modules 1–9 converges here into one coherent,
deployable product. You will not just learn pieces — you will assemble them into a system
that behaves like something you'd ship to a real team.

---

## 1. Product Definition

**Papeer** is a *research assistant* for a document corpus. A user asks a natural-language
question; Papeer retrieves the most relevant passages from an ingested knowledge base,
reranks them, reasons over them with an LLM agent that can reflect on its own answer, and
returns a grounded, cited response.

- **User:** a researcher / analyst / engineer who has a private corpus (notes, docs, data).
- **Job-to-be-done:** "Give me a fast, accurate, *cited* answer grounded only in my documents."
- **Non-goals (v1):** multi-turn memory, image/multimodal retrieval, fine-tuning, web search.

### Core value proposition
| Property | How Papeer delivers it |
|----------|------------------------|
| Accuracy | Hybrid (dense + BM25) retrieval + cross-encoder rerank |
| Grounding | Agent grades context before answering; citations in output |
| Robustness | Reflection loop re-retrieves when context is weak |
| Trust | RAGAS evaluation gate + guardrails-ready API |

---

## 2. Scope

**In scope (this module):**
- Multi-format ingestion (`.txt`, `.md`, `.csv`, `.json`, `.pdf`) from `data/samples/` + user docs
- Chunking with rich metadata (source, page, chunk index)
- Persistent ChromaDB index + JSON sidecar for BM25
- Hybrid retrieval: dense top-10 + BM25 top-10 → Reciprocal Rank Fusion → cross-encoder top-3
- LangGraph agent: retrieve → grade → generate → reflect (max 2 iterations)
- FastAPI service (`POST /ask`) with streaming-ready structure
- RAGAS evaluation on a small built-in golden set
- CLI demo (`main.py`) with per-stage timing

**Out of scope (deferred):** auth/RBAC, semantic caching (Module 11), full observability stack,
load testing, multi-tenancy. These are deliberately covered in Module 11.

---

## 3. Architecture Decisions (ADR summary)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Vector store | ChromaDB (persistent) | Zero-config, HNSW; swap to Qdrant/pgvector later |
| 2 | Sparse index | rank-bm25 over JSON sidecar | Cheap, no extra infra; complements dense for exact terms |
| 3 | Fusion | Reciprocal Rank Fusion (k=60) | Model-agnostic, robust, industry default for hybrid |
| 4 | Rerank | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precision stage; small set (top-20→top-3) is fast |
| 5 | Orchestration | LangGraph StateGraph | Explicit control flow + reflection loop, testable |
| 6 | LLM | OpenAI if key else Ollama llama3.1 | Local-first, zero-key works |
| 7 | Embeddings | sentence-transformers MiniLM (local) | Free, fast, deterministic |
| 8 | Service | FastAPI + uvicorn | Async, streaming-ready, standard for prod APIs |
| 9 | Eval | RAGAS | Standard metrics: faithfulness, relevancy, precision/recall |

---

## 4. Phased Action Plan

### Phase 1 — Ingestion (`ingestion.py`)
Load every supported file in `data/samples/` (+ optional user dir), split into chunks with
metadata, embed, persist to Chroma, and dump a JSON sidecar for BM25.
**Exit:** `uv run python code/papeer/ingestion.py` builds the index; chunk count printed.

### Phase 2 — Hybrid Retrieval + Rerank (`retrieval.py`)
Dense Chroma top-10 + BM25 top-10 → RRF → cross-encoder rerank → top-3 with scores.
**Exit:** `retrieve(question)` returns ranked passages with source + score.

### Phase 3 — Agentic Workflow (`agent.py`)
LangGraph graph: retrieve → grade → generate → reflect, max 2 iterations. Records per-stage
timings.
**Exit:** `run_agent(question)` returns answer + stage timings.

### Phase 4 — Evaluation (`evaluate.py`)
Run RAGAS over a small built-in golden set; print faithfulness / relevancy / precision / recall.
**Exit:** report printed; graceful fallback if RAGAS unavailable.

### Phase 5 — Deployment (`api.py` + `main.py`)
FastAPI `POST /ask` (streaming-ready) + CLI demo running 3 questions end-to-end with timing.
**Exit:** `main.py` prints stages, answers, per-stage latency for 3 questions.

---

## 5. Deliverables Checklist

- [ ] `code/papeer/config.py` — paths & params
- [ ] `code/papeer/ingestion.py` — multi-format ingestion + chunking + persist
- [ ] `code/papeer/retrieval.py` — hybrid + RRF + rerank
- [ ] `code/papeer/agent.py` — LangGraph agent with reflection loop
- [ ] `code/papeer/api.py` — FastAPI service (documented, needs `uv add fastapi uvicorn`)
- [ ] `code/papeer/evaluate.py` — RAGAS eval on golden set
- [ ] `code/papeer/main.py` — CLI demo, 3 questions, per-stage timing
- [ ] Index built under `data/chroma_db/papeer/`
- [ ] Golden-set evaluation report produced
- [ ] `notes.md` filled with your takeaways

---

## 6. Time Estimate

| Phase | Task | Est. |
|-------|------|------|
| 1 | Ingestion | 1.5 h |
| 2 | Hybrid retrieval + rerank | 2 h |
| 3 | Agentic workflow | 2 h |
| 4 | Evaluation | 1 h |
| 5 | Deployment + demo | 1.5 h |
|   | **Total** | **~8 h** |

---

## 7. Success Criteria (measurable)

| Metric | Target | How measured |
|--------|--------|--------------|
| Faithfulness (RAGAS) | **> 0.8** on golden set | `evaluate.py` |
| Answer relevancy | > 0.7 | `evaluate.py` |
| Context precision@k | > 0.6 | `evaluate.py` |
| Retrieval recall (top-3) | relevant doc present ≥ 4/5 golden Qs | manual + eval |
| p95 end-to-end latency | < 8 s (local Ollama) / < 3 s (OpenAI) | `main.py` timing |
| Per-stage budget | retrieve < 1 s, rerank < 0.5 s, generate < 5 s | `main.py` timing |
| Zero-key operation | full pipeline runs with local models only | run without `.env` keys |
| Graceful degradation | missing deps / no Ollama → clear error, no crash | error handling |
