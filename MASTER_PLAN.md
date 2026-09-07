# 🎯 MASTER PLAN — Advanced RAG Course (v2 Expansion)

> **Pattern: Research → Plan → Implement.** This document is the single source of truth for the
> v2 expansion. Every subagent task must trace back to a deliverable below.

## 1. Mission

Build a **complete, industry-standard, locally-runnable Advanced RAG learning system** that also
serves as a **GitHub portfolio proving subject-matter expertise** — from planning → designing →
development → testing → production deployment → live to users.

**Primary goal:** Learn RAG *by doing*, with **multiple implementation approaches per technique**
(framework-based AND from-scratch), exactly as practiced in industry.

## 2. Current State (v1 — COMPLETE ✅)

| Item | Status |
|---|---|
| 11 module folders (`01-…` … `11-…`) with `01-plan.md`, `02-learning.md`, `03-implementation.md`, `code/`, `notes.md` | ✅ Verified |
| `shared/config.py` model factories (Ollama/OpenAI/local embeddings) | ✅ |
| `pyproject.toml` (uv) with all deps | ✅ |
| Sample data (txt/csv/json/md) | ✅ |
| All code compiles; non-LLM scripts run; LLM scripts degrade gracefully | ✅ |

## 3. v2 Deliverables

### D1 — Multi-Approach Code for Every Module
Each module gets **2–3 additional runnable scripts** in `code/`:
- **Approach A** (existing `main.py`): LangChain/LangGraph framework-based (industry default).
- **Approach B**: alternative framework usage (e.g., raw ChromaDB API, fastembed, different splitter).
- **Approach C**: **from-scratch implementation** (pure Python/numpy) proving deep understanding.
- Plus **benchmark/comparison scripts** where meaningful (chunking sizes, retriever quality, cache hit rates).

### D2 — Noob → Expert Architecture Diagrams
Every module's `02-learning.md` gets a **"Diagrams: Noob → Expert"** section with ≥3 mermaid
diagrams at increasing depth:
1. **Noob level** — simple flowchart of the concept.
2. **Practitioner level** — component diagram with data types & libraries.
3. **Expert level** — sequence/state diagram showing edge cases, failure modes, performance knobs.

### D3 — GitHub Portfolio Projects (top-level `portfolio/`)
Three standalone, production-grade projects, each with its own `pyproject.toml`, tests, Docker,
CI workflow, and full lifecycle docs:

| # | Project | Proves |
|---|---|---|
| P1 | `portfolio/enterprise-knowledge-assistant/` | Production RAG **service**: FastAPI + hybrid retrieval + rerank + semantic cache + guardrails + Docker + pytest + CI + monitoring |
| P2 | `portfolio/agentic-research-agent/` | **Agentic RAG**: LangGraph plan-and-execute + reflection, multi-tool, streaming API, citations, Docker + tests |
| P3 | `portfolio/rag-evaluation-harness/` | **Eval & MLOps**: RAGAS + custom metrics, A/B pipeline comparison, regression reports, CI-gated quality |

Each includes: `README.md` (pitch + quickstart), `docs/PLANNING.md`, `docs/DESIGN.md` (mermaid
architecture), `docs/DEPLOYMENT.md` (prod runbook), `tests/`, `Dockerfile`, `docker-compose.yml`,
`.github/workflows/ci.yml`, `.env.example`.

### D4 — Agent Skills
- **Cursor**: `.cursor/rules/advance-rag-course.mdc` — always-on project guidelines.
- **Goose**: `.agents/skills/advance-rag/SKILL.md` — expert workflow skill (conventions, pitfalls, verification protocol).

### D5 — Yolo-Auto Provider Integration
`shared/config.py` supports 3-tier LLM selection:
1. `OPENAI_API_KEY` → OpenAI
2. `YOLO_AUTO_API_KEY` → Yolo-Auto (OpenAI-compatible, base `https://yolo-auto.com/v1`, model `qwen3.8-27b`, 131k ctx)
3. Fallback → Ollama local

### D6 — Documentation Upgrade
- Root `README.md`: roadmap + portfolio links + skills + provider matrix.
- Every module doc updated to reference all approaches with trade-off tables.
- Mermaid diagrams throughout (≥3 per module, ≥5 per portfolio project).

## 4. Per-Module Expansion Spec

| Module | New code files (in `code/`) | Focus |
|---|---|---|
| 01 Fundamentals | `approach_2_stepback_rag.py`, `approach_3_rag_from_scratch.py` | Step-back prompting; raw Chroma+client RAG (no LangChain) |
| 02 Doc Processing | `approach_2_semantic_chunking.py`, `approach_3_code_splitter.py`, `chunking_benchmark.py` | Embedding-similarity chunking (from scratch); LanguageTextSplitter; size/overlap benchmark |
| 03 Embeddings | `approach_2_cloud_embeddings.py`, `embedding_visualization.py` | Cloud (OpenAI/Yolo-Auto) embeddings w/ graceful fallback; PCA viz (matplotlib) |
| 04 Vector Stores | `approach_2_raw_chroma_api.py`, `approach_3_hnsw_tuning.py`, `metadata_filters_advanced.py` | Raw `chromadb.PersistentClient`; HNSW param tuning w/ timing; `$and/$or/$in/$not` |
| 05 Basic Retrieval | `approach_2_fastembed_hybrid.py`, `approach_3_rrf_from_scratch.py`, `retrieval_benchmark.py` | fastembed dense+sparse; RRF implemented in numpy; head-to-head table |
| 06 Advanced Retrieval | `approach_2_parent_doc_from_scratch.py`, `approach_3_compression_comparison.py` | Parent-doc pattern without LangChain; compression on/off quality delta |
| 07 Advanced Patterns | `rag_fusion/approach_2_from_scratch.py`, `hyde/approach_2_from_scratch.py`, `pattern_comparison.py` | Framework-free RRF fusion; framework-free HyDE; side-by-side pattern shootout |
| 08 Agentic RAG | `plan_and_execute_rag.py`, `multi_tool_agent.py`, `streaming_agent.py` | Plan-and-Execute graph; multi-tool ReAct; token streaming |
| 09 Evaluation | `approach_2_custom_metrics.py`, `approach_3_llm_judge_from_scratch.py`, `ab_pipeline_comparison.py` | Metrics without RAGAS; DIY LLM-as-judge; naive-vs-hybrid A/B |
| 10 Capstone | `tests/test_ingestion.py`, `tests/test_retrieval.py`, `deploy/Dockerfile`, `deploy/docker-compose.yml`, `deploy/DEPLOYMENT.md`, `ui/streamlit_app.py` | pytest suite; containerized deployment; live UI |
| 11 Production | `cost_optimization.py`, `caching_strategies.py`, `debugging_playbook.py`, `security_compliance.py` | Token/cost accounting; exact vs semantic cache; failure triage; PII/access-control |

## 5. Conventions (MANDATORY for all code)

1. **Windows UTF-8 guard** at top of every runnable script:
   ```python
   import sys
   if sys.platform == "win32":
       sys.stdout.reconfigure(encoding="utf-8", errors="replace")
       sys.stderr.reconfigure(encoding="utf-8", errors="replace")
   ```
2. **Shared imports**: `sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))`
   (use `parents[2]` for `code/<subdir>/` scripts) then `from shared.config import get_llm, get_embeddings`.
3. **Graceful degradation**: if Ollama/cloud unavailable, print a clear hint and exit 0 — never crash.
4. **Known API pitfalls** (LangChain 1.x era):
   - `rank_bm25.BM25Okapi` (NOT `RankBM25`)
   - Advanced retrievers: `try: from langchain_classic.retrievers import X` / `except: from langchain.retrievers import X`
   - `LLMChainExtractor.from_llm(llm)` (not `llm=` kwarg)
   - `ParentDocumentRetriever`: check `'byte_store' in ParentDocumentRetriever.model_fields`
   - `SelfQueryRetriever.from_llm(..., document_contents=..., metadata_field_info=[...])`; neutralize literal `{}` in sample text
   - `InMemoryStore` from `langchain_core.stores`
   - `JSONLoader(file_path=..., jq_schema='.', text_content=False)`
5. **Verification protocol**: `uv run python -m py_compile <file>` for ALL new files; FULLY RUN any
   script that needs no LLM/no big downloads; NEVER execute scripts that download >100MB models.
6. **Docs**: mermaid diagrams, trade-off tables, "why" explanations, noob→expert progression.
7. **Output**: `rich` console for pretty CLI output.

## 6. Execution Plan (≤6 parallel background tasks per batch)

| Batch | Tasks |
|---|---|
| 0 (main agent) | MASTER_PLAN, config.py + .env (Yolo-Auto), pyproject deps, Cursor skill, Goose skill, `uv sync` |
| A (6 agents) | Modules 01–06 expansion |
| B (6 agents) | Modules 07–11 expansion |
| C (3 agents) | Portfolio P1, P2, P3 |
| Final (main) | Full compile check, spot-run scripts, root README rewrite, status table update |

## 7. Status Tracker

| Deliverable | Status |
|---|---|
| MASTER_PLAN.md | ✅ |
| Yolo-Auto in shared/config.py + .env.example | ✅ |
| Cursor rules (`.cursor/rules/advance-rag-course.mdc`) | ✅ |
| Agent skill (`.agents/skills/advance-rag/`) | ✅ |
| Module 01–06 multi-approach expansion | ✅ verified: 15/15 scripts compiled + full-ran; diagrams 6/6 |
| Module 07–11 multi-approach expansion | ✅ verified: all scripts compile; capstone pytest 11/11 green; diagrams 5/5 |
| Portfolio P1 enterprise-knowledge-assistant | ✅ pytest 5/5 green; FastAPI app imports; Docker + deploy docs |
| Portfolio P2 agentic-research-agent | ✅ pytest 7/7 green; FastAPI app imports; route-shadowing bug fixed; Docker + deploy docs |
| Portfolio P3 rag-evaluation-harness | ✅ pytest 9/9 green; CLI run+gate e2e; golden dataset + report |
| Root README v2 + final verification | ✅ 11/11 modules marked done + portfolio section |
