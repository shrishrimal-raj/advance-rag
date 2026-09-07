# 🎯 Week 05 — Evals & Observability

> Status: ⏳ PENDING (Batch — see `../CHECKPOINT.md`). This is the planning doc; learning + implementation + code land when this week's batch runs.

## Objective
An automated eval and tracing harness that logs execution traces and blocks CI/CD deploys on quality regression.

## What You'll Learn
- End-to-End Tracing: capturing LLM inputs, tool execution & retrieval steps
- Unit Economics: token cost and latency breakdown per request
- Golden Datasets & LLM-as-a-Judge Scoring
- Evaluating Agents, RAG Pipelines & Tool Call Trajectories

## Tools / Stack
- Langfuse
- OpenTelemetry
- RAGAS
- DeepEval

## Weekly Build
**The Regression & Telemetry Gate** — An automated eval and tracing harness that logs execution traces and blocks CI/CD deploys on quality regression.

**Outcome:** Trace production execution step-by-step and make underperforming prompts physically unable to ship.

## Deliverables (this week)
- [ ] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [ ] `03-implementation.md` — step-by-step build guide
- [ ] `code/main.py` — framework-based end-to-end build
- [ ] `code/approach_2_*.py` — alternative framework usage
- [ ] `code/approach_3_*.py` — from-scratch implementation
- [ ] `code/*_benchmark.py` — head-to-head comparison (where meaningful)
- [ ] All scripts: Windows UTF-8 guard + graceful degradation + `py_compile` clean

## Prerequisites
- Previous week(s) complete.
- `uv sync` done in `ai-engineering/`; `YOLO_AUTO_API_KEY` set (cloud LLM). No Ollama (8 GB laptop).

## Time Estimate
~4–6 hours hands-on.
