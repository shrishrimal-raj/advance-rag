# 🎯 Week 08 — Agentic System Design & Reliability Engineering

> Status: ⏳ PENDING (Batch — see `../CHECKPOINT.md`). This is the planning doc; learning + implementation + code land when this week's batch runs.

## Objective
A secure, scalable dual-agent supervisor system that survives a live provider-outage recovery drill.

## What You'll Learn
- Building Agentic Systems for Scale
- Design Trade-offs
- Multi-Agent Maker-Checker Topologies
- AI Security (Prompt Injection, PII Redaction)
- Fallbacks, Semantic Caching & Rate Limiting

## Tools / Stack
- Multi-Agent
- System Design

## Weekly Build
**The Dual-Agent Supervisor** — A secure, scalable dual-agent supervisor system that survives a live provider-outage recovery drill.

**Outcome:** Learn to attack your own system, patch holes, and implement graceful failure mechanisms.

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
