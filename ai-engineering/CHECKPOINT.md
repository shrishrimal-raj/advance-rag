# ✅ CHECKPOINT.md — SDE → AI Engineer Course Progress Tracker

> **Live tracker.** Single source of truth for what is DONE vs LEFT. Update after every batch/wave.
> Legend: ✅ done+verified by main · 🔄 in-progress/file exists, unverified · ⏳ pending · ❌ failed/blocked
>
> **Last updated:** 2026-09-07 by main agent — **Batch 0 COMPLETE**: infra (plan/checkpoint/README/config/skeleton) + **Week 1 fully built & verified** (4/4 compile; from-scratch tokenizer+attention runs offline; AI Gateway selftest streams via Yolo-Auto w/ live cost tracking). Next: Batch A (Weeks 2–4).

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
| D1 | 10 weekly modules, multi-approach code | 🔄 | main | Skeleton (Wk2–10) created; **Week 1 ✅ built+verified**; Wk2–10 pending |
| D2 | Noob→Expert mermaid diagrams in every `02-learning.md` | ⏳ | main | With each module |
| D3a | `portfolio/ai-gateway-service/` (P1) | ⏳ | main | Batch D |
| D3b | `portfolio/enterprise-search-service/` (P2) | ⏳ | main | Batch D |
| D3c | `portfolio/regression-telemetry-gate/` (P3) | ⏳ | main | Batch D |
| D4 | Agent skills (Cursor + Goose) | ⏳ | main | Final batch |
| D6b | Root-level: create **master** branch + push to origin | ⏳ | main | After Batch 0 minimum |

---

## 2. Per-Week Detail (D1 code + D2 diagrams)

| Wk | Folder | Docs | Code (main/a2/a3/bench) | Diagrams | Status |
|----|--------|------|--------------------------|----------|--------|
| 1 | `01-python-llm-fundamentals/` | ✅ | ✅ | ✅ (6 mermaid) | ✅ **COMPLETE** — 4/4 compile; approach_3 runs offline; gateway selftest live via Yolo-Auto |
| 2 | `02-rag-foundations/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch A) |
| 3 | `03-enterprise-rag-pipelines/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch A) |
| 4 | `04-ai-agents-state-machines/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch A) |
| 5 | `05-evals-observability/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch B) |
| 6 | `06-mcp-context-multi-agent/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch B) |
| 7 | `07-ml-fine-tuning/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch B) |
| 8 | `08-agentic-system-design/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch C) |
| 9 | `09-multimodal-ai/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch C) |
| 10 | `10-capstone-finale/` | ⏳ | ⏳ | ⏳ | ⏳ pending (Batch C) |

---

## 3. Portfolio Projects (D3) — pending (Batch D)

Spec per project: `README.md`, `pyproject.toml`, `docs/PLANNING.md`, `docs/DESIGN.md` (≥5 mermaid), `docs/DEPLOYMENT.md`, `tests/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `.env.example`, working app code.

- P1 `portfolio/ai-gateway-service/` — ⏳
- P2 `portfolio/enterprise-search-service/` — ⏳
- P3 `portfolio/regression-telemetry-gate/` — ⏳

---

## 4. Batch Log

| # | Batch | Waves (≤3–4 concurrent) | Status | Started | Finished | Result |
|---|-------|--------------------------|--------|---------|----------|--------|
| 0 | Infra + Week 1 (plan, checkpoint, README, config, skeleton, Wk1 full) | main only | ✅ | 2026-09-07 | 2026-09-07 | Wk1 verified: 4/4 compile, offline demo OK, live selftest OK |
| A | Weeks 2, 3, 4 | 3 agents | ⏳ | — | — | — |
| B | Weeks 5, 6, 7 | 3 agents | ⏳ | — | — | — |
| C | Weeks 8, 9, 10 | 3 agents | ⏳ | — | — | — |
| D | Portfolio P1, P2, P3 | 3 agents | ⏳ | — | — | — |
| F | Final: compile-all, spot-run, README final, skills, verify, push master | main only | ⏳ | — | — | — |

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
