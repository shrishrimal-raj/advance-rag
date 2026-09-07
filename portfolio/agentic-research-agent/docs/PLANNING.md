# Planning — Agentic Research Agent

## Problem Statement

A plain RAG pipeline (retrieve → stuff into prompt → answer) fails on research-style
questions that need **multiple, heterogeneous evidence-gathering steps**: a semantic
search for concepts, an exact metadata lookup for a specific document or topic, and
occasionally a numeric computation. A single retrieval call either misses the right
slice of the corpus or drowns the model in irrelevant chunks, and there is no
mechanism to notice "the evidence is insufficient" and search again.

This project builds an **agentic** alternative: a LangGraph plan-and-execute loop with
a reflection (critic) pass, three tools (`vector_search`, `metadata_filter_search`,
`calculator`), deterministic citation numbering, and a streaming SSE API — all backed
by a **local** Chroma vector store and local MiniLM embeddings so retrieval never
depends on the network.

## Goals

- Multi-step research: decompose a question into 2–4 tool steps, execute them, and
  synthesize an answer grounded only in gathered observations.
- Self-correction: a reflection node scores evidence confidence (0–100) and triggers
  **at most one** revision pass when confidence < `REFLECT_CONFIDENCE_THRESHOLD`
  (default 60), bounding LLM cost.
- Trustworthy output: citations are numbered from actual observation sources and
  sanitized so the answer can never contain dangling `[n]` markers.
- Safe tool surface: the calculator evaluates pure arithmetic via an AST allowlist
  (no names, no calls) — safe to expose to an LLM-driven loop.
- Streaming UX: `POST /ask/stream` emits server-sent events
  (`phase` → `token` → `done`) so clients see planning progress before the answer.
- Offline-testable: the full test suite runs with no LLM key and no network
  (ephemeral Chroma + fake embeddings).

## Non-Goals

- No web crawling / live web search — the corpus is a fixed local Chroma collection
  (`research_docs`).
- No multi-agent debate, no parallel sub-agents, no memory across questions.
- No auth, multi-tenancy, or per-user state in this version.
- No GPU/local LLM requirement — the LLM is a cloud OpenAI-compatible endpoint
  (Yolo-Auto by default, OpenAI if configured, Ollama as last resort).
- No re-ingestion pipeline in the repo yet — corpus ingestion is a one-time manual
  step that fills the `research_docs` collection (see Risks).

## Requirements

| ID | Requirement | Where implemented | Priority |
|----|-------------|-------------------|----------|
| R1 | Planner decomposes question into 2–4 JSON tool steps; falls back to one broad `vector_search` if LLM output is unparseable | `agent/graph.py::planner` | Must |
| R2 | Executor runs each planned step through `ResearchTools.run`; a failing tool records `error:` in the observation instead of killing the run | `agent/graph.py::executor` | Must |
| R3 | Reflector returns `{confidence, critique, missing}`; low confidence + not-yet-revised appends up to 3 gap-filling `vector_search` steps exactly once (`revision_done` guard) | `agent/graph.py::reflector`, `route_after_reflection` | Must |
| R4 | Synthesizer answers using only numbered sources; `sanitize_citations` drops out-of-range `[n]` markers | `agent/graph.py::synthesizer`, `sanitize_citations` | Must |
| R5 | `vector_search` returns top-k `{text, source, score}` from Chroma cosine space | `agent/tools.py::ResearchTools.vector_search` | Must |
| R6 | `metadata_filter_search` parses `field=value`, filters via Chroma `where`, ranks matches by embedding similarity to the value | `agent/tools.py::ResearchTools.metadata_filter_search` | Must |
| R7 | `calculator` safely evaluates arithmetic via AST allowlist; anything else returns an `error:` string, never raises | `agent/tools.py::calculate` | Must |
| R8 | `GET /health` reports LLM status (`configured`/`missing`) and corpus size | `api/routes.py::health` | Must |
| R9 | `POST /ask` returns answer, citations, plan, reflections; 503 with actionable detail on failure | `api/routes.py::ask_impl` | Must |
| R10 | `POST /ask/stream` streams SSE events: `phase` (planning/executing/reflecting/synthesizing), `token`, `done`, `error` | `api/routes.py::ask_stream`, `agent/graph.py::stream_research` | Should |
| R11 | 3-tier LLM selection: OpenAI → Yolo-Auto (OpenAI-compatible) → Ollama; cached singleton | `agent/graph.py::_default_llm` | Must |
| R12 | Offline test suite: calculator safety, tool dispatch, vector + metadata search on ephemeral Chroma | `tests/test_tools.py` | Must |
| R13 | LLM JSON parsing tolerates markdown fences and prose-wrapped JSON | `agent/graph.py::_extract_json` | Should |

## Milestones

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| M1 | Core graph: planner → executor → reflector → synthesizer with conditional revision edge | Done | `agent/graph.py`, `agent/state.py`, `agent/prompts.py` |
| M2 | Tool layer: Chroma vector + metadata-filter search, safe calculator, uniform `run()` dispatch | Done | `agent/tools.py` |
| M3 | Streaming API: FastAPI app, `/health`, `/ask`, `/ask/stream` (SSE), lazy LLM/tools wiring | Done | `main.py`, `api/routes.py` |
| M4 | Offline test suite passing without keys/network | Done | `tests/test_tools.py` (7 tests) |
| M5 | Packaging & ops: Dockerfile, docker-compose, CI workflow, deployment docs | In progress | `Dockerfile`, `docs/DEPLOYMENT.md` exist; compose + CI added here |
| M6 | Corpus ingestion script + sample docs | Not started | `scripts/` and `data/sample_docs/` are empty placeholders; ingestion currently manual |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Reflection loop multiplies LLM cost per question | High | Medium | Hard cap: exactly one revision pass (`revision_done`); threshold configurable via `REFLECT_CONFIDENCE_THRESHOLD`; rate-limit at proxy (see DEPLOYMENT.md) |
| LLM emits malformed JSON (plan/reflection) | Medium | Medium | `_extract_json` strips fences and scans for first balanced bracket pair; planner falls back to a single broad `vector_search` step |
| Empty corpus at serve time | Medium | High | `/health` exposes `corpus_docs`; empty collection makes searches return `[]` rather than crash; ingestion documented as pre-serve step |
| Missing API key at startup | Low | Low | App still boots; `/health` reports `llm: missing`; `/ask` returns 503 with a message pointing at `.env` |
| Calculator abuse by LLM-generated input | Low | High | AST allowlist: numeric constants, `+ - * / // % **`, unary `+/-` only; names/calls/subscripts raise → returned as `error:` string |
| Citation hallucination (`[9]` with 4 sources) | Medium | Medium | `build_citations` numbers real sources; `sanitize_citations` strips any marker outside `1..N` before response/stream completion |
| Embedding model download on first run (~90 MB MiniLM) | Low | Low | Cached after first use; pinned via `LOCAL_EMBEDDING_MODEL`; below the 100 MB download cap |

## Success Metrics

- **Correctness**: 100% of offline tests pass in CI (`uv run pytest -v`) with no network.
- **Safety**: calculator rejects all non-arithmetic input (covered by `test_calculate_rejects_unsafe`); zero dangling citation markers in any returned answer (enforced by `sanitize_citations`).
- **Cost bound**: ≤ 5 LLM calls per question (planner + reflector + synthesizer = 3 baseline; +2 max on the single revision pass).
- **Resilience**: app boots and serves `/health` with no API key configured; a failing tool degrades to an `error:` observation instead of a 500.
- **Latency**: streaming endpoint emits the first `phase` event before any synthesis token, so perceived wait starts immediately.
- **Ops**: `docker compose up` reproduces the dev environment; CI fails the build on any test failure.
