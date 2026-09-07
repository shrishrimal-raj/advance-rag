# 🎯 MASTER PLAN — SDE → AI Engineer Course (10-Week Cohort Recreation)

> **Pattern: Research → Plan → Implement.** Single source of truth for building the
> "From SDE To AI Engineer" course as a local, industry-standard learning system + GitHub
> portfolio. Every subagent task must trace back to a deliverable below.

## 0. Sources of Truth
- **Task content:** `ai-engineering/Task.md` — Coding Shuttle "SDE → AI Engineer" 10-week cohort outline (weeks, weekly builds, tools, outcomes).
- **Reference pattern:** root `MASTER_PLAN.md` / `CHECKPOINT.md` / `README.md` — the *completed* Advanced RAG course. We mirror its structure, conventions, and verification protocol.

## 1. Mission
Recreate the 10-week "SDE → AI Engineer" curriculum as a **locally-runnable learning system**
under `ai-engineering/`, with **multiple implementation approaches per technique**
(framework-based AND from-scratch), **noob→expert architecture diagrams**, and **production
portfolio projects** — proving end-to-end subject-matter expertise:
planning → design → development → testing → deployment.

## 2. HARD CONSTRAINTS (8 GB low-power laptop)
- **NEVER run/pull Ollama locally.**
- **LLM = Yolo-Auto cloud** (`YOLO_AUTO_API_KEY` in root `.env`, verified working). Local MiniLM embeddings OK (cached, CPU-light).
- **Minimize full-pipeline LLM runs** (1–2 per module; each 2–4 min).
- **Heavy workloads** (fine-tuning, vision, voice) = ship CODE + docs + cloud/GPU runbook; do **NOT** run locally. Never download >100 MB models locally.
- **Subagents:** ≤3–4 concurrent (provider ~128K ctx each). Self-contained briefs. **No commit/push by subagents**; main verifies all output.

## 3. Deliverables
| ID | Deliverable | Notes |
|----|-------------|-------|
| D1 | 10 weekly modules, multi-approach code | Each: `01-plan.md`, `02-learning.md`, `03-implementation.md`, `code/` (main + approach_2 + approach_3 + benchmark), `notes.md` |
| D2 | Noob→Expert diagrams | ≥5 mermaid blocks per `02-learning.md` |
| D3 | Portfolio projects (`ai-engineering/portfolio/`) | **10 projects** = 1 Capstone + 9 weekly builds per Task.md (each own pyproject/tests/Docker/CI/docs) |
| D4 | Agent skills | Cursor rule + Goose skill for this course |
| D5 | Provider integration | Reuse root 3-tier config (OpenAI → Yolo-Auto → Ollama) |
| D6 | Docs upgrade | Course README roadmap + provider matrix + structure |

## 4. Per-Week Spec

| Wk | Folder | Learn | Tools | Weekly Build | Code approaches (main / a2 / a3 / bench) |
|----|--------|-------|-------|--------------|------------------------------------------|
| 1 | `01-python-llm-fundamentals/` | Python async, APIs & streaming; transformer internals, tokenization, vectorization, attention; LLM lifecycle & model tiering | Python, FastAPI, Transformers | **The AI Gateway** — streaming non-blocking chat backend w/ token usage + cost tracking | main=FastAPI gateway · a2=raw httpx OpenAI-compat streaming · a3=from-scratch tokenizer+attention (numpy) · bench=model-tier latency/cost |
| 2 | `02-rag-foundations/` | Embeddings & vector geometry; multi-tenant isolation; vector DBs & indexing; doc parsing & chunking | pgvector, Embeddings | **The Knowledge Engine** — context-aware RAG over local vector store | main=LangChain RAG · a2=raw vector DB API · a3=from-scratch cosine retrieval (numpy) · bench=chunking strategies |
| 3 | `03-enterprise-rag-pipelines/` | Hybrid search (semantic+keyword); cross-encoder rerank; query expansion & graph RAG basics | BM25, RRF | **The Enterprise Search Engine** — multi-tenant re-ranked hybrid search | main=hybrid+rerank · a2=fastembed hybrid · a3=RRF from scratch (numpy) · bench=naive vs hybrid vs reranked |
| 4 | `04-ai-agents-state-machines/` | LLM vs agent vs multi-agent; ReAct loop & reliable tool calling; chaining/orchestration/routing; LangGraph state machines | ReAct, Multi-Agent, LangGraph | **The Persistent Operator** — tool-calling agent w/ Redis persistent memory, human-approval pause/resume | main=ReAct+tools (LangGraph) · a2=multi-tool agent · a3=from-scratch ReAct loop · +Redis memory |
| 5 | `05-evals-observability/` | E2E tracing; unit economics (cost/latency); golden datasets & LLM-as-judge; eval agents/RAG/tool trajectories | Langfuse, OpenTelemetry, RAGAS, DeepEval | **The Regression & Telemetry Gate** — eval+tracing harness, blocks CI/CD on regression | main=RAGAS harness · a2=custom metrics (no RAGAS) · a3=DIY LLM-as-judge · +CI gate script |
| 6 | `06-mcp-context-multi-agent/` | Standardize tools via MCP; custom MCP servers & clients; context engineering; memory in agents; multi-agent orchestration | MCP, Security, Context Eng, Memory | **The Audited Tool Bridge** — wrap internal systems into audited tools, expose via multi-tool agent bridge | main=custom MCP server (FastMCP) · a2=MCP client consuming server · a3=multi-agent orchestration (CrewAI/LangGraph supervisor) |
| 7 | `07-ml-fine-tuning/` | Core ML math & fine-tuning; LoRA/QLoRA & dataset engineering; FT vs prompting vs RAG; quantization & local serving | LoRA, Quantization, LLM Opt | **The Specialist Model** — fine-tune open-weight model, benchmark vs frontier APIs on dashboard | main=LoRA/QLoRA script (**code+runbook, NOT local-run on 8GB**) · a2=dataset engineering · a3=quantization comparison · bench=head-to-head Streamlit dashboard |
| 8 | `08-agentic-system-design/` | Agentic systems for scale; design trade-offs; maker-checker topologies; AI security (prompt injection, PII redaction); fallbacks, semantic caching, rate limiting | Multi-Agent, System Design | **The Dual-Agent Supervisor** — secure dual-agent surviving live provider-outage recovery drill | main=dual-agent maker-checker (LangGraph) · a2=failover/fallback logic · a3=prompt-injection defense + PII redaction · +chaos drill |
| 9 | `09-multimodal-ai/` | Multimodal architecture; low-latency multimodal; multimodal RAG; capstone scoping | Vision, Voice AI, Video | **The Multimodal RAG Engine** — vision+voice channels, structured extraction, cost-per-ticket dashboard | main=multimodal RAG (vision+text) · a2=voice channel (Whisper) · a3=structured extraction · ⚠️ heavy models degrade gracefully |
| 10 | `10-capstone-finale/` | Recap; interview prep; defending design decisions; build capstone E2E; feedback | Interview Prep, Capstone, Evals, Monitoring | **The Finale Deploy** — enterprise-grade MVP w/ agents, evals & observability dashboard | capstone app (integrates prior weeks) · Docker+AWS deploy · observability dashboard · demo-day deck |

### D3 — Portfolio Projects (1 Capstone + 9 weekly builds, per Task.md)

Task.md ships **10 production-grade projects** ("1 Capstone + 9 Projects Built From Scratch"). Each is an industry-standard, portfolio-ready service: own `pyproject.toml`, offline-testable core, FastAPI app, `tests/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `.env.example`, and `docs/{PLANNING,DESIGN(≥5 mermaid),DEPLOYMENT}.md`. Real integrations (pgvector/Redis/Pinecone/CrewAI/Whisper/W&B) are wired as guarded, swappable dependencies; the testable core runs fully offline so CI is green with no secrets or heavy models.

| # | Project (dir) | Task build | Tech stack | Source wk |
|---|---------------|-----------|-----------|-----------|
| 1 | `portfolio/capstone-enterprise-mvp/` | Capstone Project | Docker·AWS·Observability | Wk10 |
| 2 | `portfolio/ai-gateway/` | The AI Gateway | FastAPI·Streaming·Cost Tracking | Wk1 |
| 3 | `portfolio/knowledge-engine/` | The Knowledge Engine | RAG·pgvector·LangChain | Wk2 |
| 4 | `portfolio/enterprise-search-engine/` | The Enterprise Search Engine | Hybrid·Multi-Tenant·Pinecone | Wk3 |
| 5 | `portfolio/persistent-operator/` | The Persistent Operator | LangGraph·Redis·Tool Calling | Wk4 |
| 6 | `portfolio/regression-gate/` | The Regression Gate | LangSmith·RAGAS·CI/CD | Wk5 |
| 7 | `portfolio/audited-tool-bridge/` | The Audited Tool Bridge | MCP·CrewAI·Docker | Wk6 |
| 8 | `portfolio/specialist-model/` | The Specialist Model | LoRA/QLoRA·W&B | Wk7 |
| 9 | `portfolio/dual-agent-supervisor/` | The Dual-Agent Supervisor | Multi-Agent·Reliability·Failover | Wk8 |
| 10 | `portfolio/multimodal-rag-engine/` | The Multimodal RAG Engine | Vision·Voice AI·Live Dashboards | Wk9 |

Each includes: `README.md`, `pyproject.toml`, `docs/PLANNING.md`, `docs/DESIGN.md` (≥5 mermaid), `docs/DEPLOYMENT.md`, `tests/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `.env.example`.

## 5. Conventions (MANDATORY for all code)
1. **Windows UTF-8 guard** at top of every runnable script:
   ```python
   import sys
   if sys.platform == "win32":
       sys.stdout.reconfigure(encoding="utf-8", errors="replace")
       sys.stderr.reconfigure(encoding="utf-8", errors="replace")
   ```
2. **Shared imports:** for a script at `<week>/code/script.py` use `sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))` (use `parents[3]` for `<week>/code/<subdir>/script.py`) then `from shared.config import get_llm, get_embeddings`. Standalone scripts that read `.env` directly load `parents[3]/.env` (repo root, holds the key) then `parents[2]/.env` (course-local).
3. **Graceful degradation:** if LLM unavailable, print a clear hint and exit 0 — never crash.
4. **Known API pitfalls** (LangChain 1.x era): `rank_bm25.BM25Okapi`; `langchain_classic` fallback imports; `LLMChainExtractor.from_llm(llm)`; `ParentDocumentRetriever` byte_store field check; SelfQuery neutralize literal `{}`; `InMemoryStore` from `langchain_core.stores`; `JSONLoader(file_path=..., jq_schema='.', text_content=False)`.
5. **Verification protocol:** `uv run python -m py_compile <file>` for ALL new files; FULLY RUN any script needing no LLM/no big downloads; NEVER download >100 MB models; minimize full-pipeline LLM runs.
6. **Docs:** mermaid diagrams, trade-off tables, "why" explanations, noob→expert progression.
7. **Output:** `rich` console for pretty CLI output.
8. **8 GB rule:** no Ollama; cloud LLM via Yolo-Auto; heavy workloads (FT/vision/voice) = code + docs, not local runs.

## 6. Execution Plan (batches, ≤3–4 concurrent subagents)
| Batch | Tasks | Mode |
|-------|-------|------|
| 0 | MASTER_PLAN, CHECKPOINT, README, pyproject, shared/config, folder skeleton, **Week 1 full** | main |
| A | Weeks 2, 3, 4 (RAG foundations → enterprise RAG → agents) | ≤3 subagents |
| B | Weeks 5, 6, 7 (evals → MCP/multi-agent → fine-tuning) | ≤3 subagents |
| C | Weeks 8, 9, 10 (system design → multimodal → capstone) | ≤3 subagents |
| D | Portfolio: all 10 Task.md projects | main (sequential) |
| Final | Full compile sweep, spot-runs, README final, skills, verify, **push master** | main |

Each batch updates `CHECKPOINT.md`. Main verifies all subagent output before marking ✅.

## 7. Status Tracker
See **`CHECKPOINT.md`** (live, single source of truth for DONE vs LEFT).
