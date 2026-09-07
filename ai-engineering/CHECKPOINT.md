# ✅ CHECKPOINT.md — SDE → AI Engineer Course Progress Tracker

> **Live tracker.** Single source of truth for what is DONE vs LEFT. Update after every batch/wave.
> Legend: ✅ done+verified by main · 🔄 in-progress/file exists, unverified · ⏳ pending · ❌ failed/blocked
>
> **Last updated:** 2026-09-07 by main agent — **Batches 0 + A + B + C + D COMPLETE**: infra + **all 10 weeks built & verified** + **all 3 portfolio projects built & tested** (offline pytest suites green). Next: Batch F (agent skills + final verify + push).

> ⚠️ **HARD CONSTRAINT (user):** 8 GB low-power laptop — NEVER run/pull Ollama locally. LLM path = Yolo-Auto cloud (key live in root `.env`, verified). Local MiniLM embeddings OK. Minimize full-pipeline LLM runs (1–2/module). Heavy workloads (fine-tune/vision/voice) = code + docs, NOT local runs.

> 📌 **Interpretation (confirm if wrong):** Task = build the "SDE → AI Engineer" 10-week course under `ai-engineering/`, mirroring the completed advance-rag pattern (root `MASTER_PLAN.md`/`CHECKPOINT.md`). Deliverable = full course + portfolio, pushed to a new **master** branch. Multi-session; resume from this file.

---

## 1. Deliverable Overview (MASTER_PLAN §3)

| ID | Deliverable | Status | Verified by | Notes |
|----|-------------|--------|-------------|-------|
| D0 | `ai-engineering/MASTER_PLAN.md` | ✅ | main | Full plan incl. per-week spec §4, conventions §5, batches §6 |
| D0b | `ai-engineering/CHECKPOINT.md` | ✅ | main | This file |
| D6a | `ai-engineering/README.md` (course overview + roadmap) | ✅ | main | Roadmap, provider matrix, structure, portfolio section |
| D5 | `ai-engineering/shared/config.py` + `.env.example` | ✅ | main | 3-tier provider; loads root `.env` (verified key); selftest confirmed |
| D1 | 10 weekly modules, multi-approach code | ✅ | main | **All 10 weeks built+verified** |
| D2 | Noob→Expert mermaid diagrams in every `02-learning.md` | ✅ | main | All 10 weeks ✅ (≥5 each) |
| D3a | `portfolio/ai-gateway-service/` (P1) | ✅ | main | 6/6 offline tests pass; gateway retry/fallback/rate-limit/log verified |
| D3b | `portfolio/enterprise-search-service/` (P2) | ✅ | main | 6/6 offline tests pass; BM25+dense+RRF+rerank verified |
| D3c | `portfolio/regression-telemetry-gate/` (P3) | ✅ | main | 6/6 offline tests pass; evals+baseline+regression+telemetry verified |
| D4 | Agent skills (Cursor + Goose) | ✅ | main | Cursor rule (`.cursor/rules/*.mdc`) + Goose skill (`.goose/skills/*/SKILL.md`) encoding course conventions |
| D6b | Root-level: create **master** branch + push to origin | ⏳ | main | After Batch 0 minimum |

---

## 2. Per-Week Detail (D1 code + D2 diagrams)

| Wk | Folder | Docs | Code (main/a2/a3/bench) | Diagrams | Status |
|----|--------|------|--------------------------|----------|--------|
| 1 | `01-python-llm-fundamentals/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; approach_3 runs offline; gateway selftest live via Yolo-Auto |
| 2 | `02-rag-foundations/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; numpy cosine + raw Chroma offline OK; live RAG selftest grounded via Yolo-Auto |
| 3 | `03-enterprise-rag-pipelines/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; RRF scratch + hybrid engine + benchmark offline OK (hybrid ranks ERR-4042 #1) |
| 4 | `04-ai-agents-state-machines/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; FSM+HITL & reach benchmark offline OK; LangGraph agent + raw ReAct selftests call tools correctly via Yolo-Auto |
| 5 | `05-evals-observability/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; gate PASS + regression detection offline; RAGAS dry-run OK (metrics need `llm=`) |
| 6 | `06-mcp-context-multi-agent/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; MCP handshake + context mgr + benchmark offline OK; LangGraph multi-agent selftest cites [r1-r4] via Yolo-Auto |
| 7 | `07-ml-fine-tuning/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; toy PyTorch loop converges (99%); dataset prep + decision matrix offline OK; HF+PEFT pipeline dry-run safe (heavy, no local train) |
| 8 | `08-agentic-system-design/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; function-calling + planner/memory + patterns matrix offline OK; autonomous agent selftest: search→fetch→answer in 3 steps via Yolo-Auto |
| 9 | `09-multimodal-ai/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; CLIP matcher + speech VAD + fusion matrix offline OK; doc pipeline selftest structures invoice JSON via Yolo-Auto |
| 10 | `10-capstone-finale/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 5/5 compile; prod-readiness + monitoring + deployment matrix offline OK; integrated platform selftest: retrieve→agent→eval(faithfulness 1.0)→trace via Yolo-Auto |

---

## 3. Portfolio Projects (D3) — COMPLETE (Batch D)

Spec per project: `README.md`, `pyproject.toml`, `docs/PLANNING.md`, `docs/DESIGN.md` (≥5 mermaid), `docs/DEPLOYMENT.md`, `tests/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `.env.example`, working app code. All delivered.

- P1 `portfolio/ai-gateway-service/` — ✅ 6/6 tests; provider routing + retry/fallback + token-bucket rate limit + structured logging; FastAPI `/health` `/ready` `/complete`
- P2 `portfolio/enterprise-search-service/` — ✅ 6/6 tests; hybrid BM25 + dense cosine + RRF + rerank; FastAPI `/health` `/ready` `/search`
- P3 `portfolio/regression-telemetry-gate/` — ✅ 6/6 tests; pluggable scorer + baseline compare + regression detect + telemetry emit; FastAPI `/health` `/ready` `/gate`

---

## 4. Batch Log

| # | Batch | Waves (≤3–4 concurrent) | Status | Started | Finished | Result |
|---|-------|--------------------------|--------|---------|----------|--------|
| 0 | Infra + Week 1 (plan, checkpoint, README, config, skeleton, Wk1 full) | main only | ✅ | 2026-09-07 | 2026-09-07 | Wk1 verified: 4/4 compile, offline demo OK, live selftest OK |
| A | Weeks 2, 3, 4 | main (sequential) | ✅ | 2026-09-07 | 2026-09-07 | Wk2/3/4 verified: compile clean, offline demos pass, live RAG+agent selftests OK |
| B | Weeks 5, 6, 7 | main (sequential) | ✅ | 2026-09-07 | 2026-09-07 | Wk5/6/7 verified: compile clean, offline demos pass, live multi-agent selftest OK |
| C | Weeks 8, 9, 10 | main (sequential) | ✅ | 2026-09-07 | 2026-09-07 | Wk8/9/10 verified: compile clean, offline demos pass, live platform selftest OK |
| D | Portfolio P1, P2, P3 | main (sequential) | ✅ | 2026-09-07 | 2026-09-07 | P1/P2/P3 verified: compile clean, 6/6 offline pytest each, FastAPI wrappers import-safe |
| F | Final: compile-all, spot-run, README final, skills, verify, push master | main only | 🔄 | 2026-09-07 | — | skills ✅; full compile sweep **58/58 OK**; fixed Wk7 unclosed-paren (re-ran → converges 99%); push **deferred per user** ("don't waste time on git") |

**Constraints honored:** ≤3–4 concurrent coding subagents (provider ~128K ctx); self-contained briefs; no commit/push by subagents; main verifies all output.

---

## 5. Resume Instructions (for next session)
1. Read this file top-to-bottom. Find the first 🔄/⏳ row in §2 or §4.
2. Continue that batch. For subagent batches, dispatch ≤3–4 concurrent with self-contained briefs (see MASTER_PLAN §4 per-week spec + §5 conventions).
3. After each wave: main verifies (py_compile all new files; run no-LLM scripts; ≤1–2 LLM smoke tests), updates §2/§4, commits.
4. Git: work happens on **master**. Commit after each verified wave. `git push origin master`.
5. Never run Ollama; never download >100 MB models; keep full-pipeline LLM runs minimal.

---

## 6. Conventions Reminder (full text in MASTER_PLAN §5)
1. Windows UTF-8 guard at top of every runnable script.
2. `sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))` then `from shared.config import get_llm, get_embeddings` (`parents[2]` for `code/<subdir>/`).
3. Graceful degradation: no LLM → clear hint + exit 0, never crash.
4. API pitfalls: `rank_bm25.BM25Okapi`; `langchain_classic` fallback; `LLMChainExtractor.from_llm(llm)`; ParentDocument byte_store check; SelfQuery neutralize `{}`; `InMemoryStore` from `langchain_core.stores`; `JSONLoader(file_path=..., jq_schema='.', text_content=False)`.
5. Verify: `uv run python -m py_compile` ALL new files; FULLY RUN no-LLM/no-big-download scripts; NEVER download >100 MB models.
6. Docs: mermaid, trade-off tables, noob→expert progression. `rich` console for CLI.
