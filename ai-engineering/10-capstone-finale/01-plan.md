# 🎯 Week 10 — Capstone Finale

> Status: ✅ BUILT (see `../CHECKPOINT.md`).

## Objective
Integrate all components into a production-ready system.

## What You'll Learn
- System Integration & Architecture
- Production Readiness & Deployment
- Monitoring & Maintenance
- Performance Optimization

## Tools / Stack
- **FastAPI** · **Docker** · **LangGraph** (installed) · **shared.config** (cloud LLM + local embeddings)

## Weekly Build
**The Enterprise AI Platform** — A complete, production-ready AI system combining all learned components: RAG retrieval + an agent + evaluation + observability, containerized and API-served.

**Outcome:** Ship a portfolio-grade system that proves you can integrate everything you've learned.

## Approach details
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | The integrated platform: retrieve -> agent -> answer -> evaluate -> trace | `--selftest` ≈ 2–3 LLM calls |
| `code/approach_2_production_readiness.py` | Config validation, health/readiness probes, graceful degradation, structured logging | **fully offline** |
| `code/approach_3_monitoring_maintenance.py` | Latency/error metrics + drift detection over a run batch | **fully offline** |
| `code/benchmark_deployment_strategies.py` | Batch vs streaming, on-prem vs cloud decision matrix | **fully offline** |

### Deploy artifacts (static, no run needed here)
- `code/app.py` — thin FastAPI wrapper exposing `/health` + `/ask`
- `Dockerfile`, `docker-compose.yml`, `.env.example`

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` · `approach_2_production_readiness.py` · `approach_3_monitoring_maintenance.py` · `benchmark_deployment_strategies.py`
- [x] Deploy artifacts: `code/app.py`, `Dockerfile`, `docker-compose.yml`, `.env.example`
- [x] All scripts: UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four code files (+ `app.py`).
- Run `approach_2` + `approach_3` + `benchmark` (offline, self-check asserts).
- Run `main.py --selftest` (bounded cloud LLM integration).

## Prerequisites
- Weeks 1–9 complete. No Ollama; no >100 MB downloads.

## Time Estimate
~6–8 hours hands-on.
