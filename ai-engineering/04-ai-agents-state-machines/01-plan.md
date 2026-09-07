# 🎯 Week 04 — AI Agents & State Machines

> Status: ⏳ PENDING (Batch — see `../CHECKPOINT.md`). This is the planning doc; learning + implementation + code land when this week's batch runs.

## Objective
An autonomous, tool-calling agent that takes real actions and remembers customer context via Redis-backed persistent memory.

## What You'll Learn
- LLM vs Agent vs Multiple Agents
- The ReAct Loop & Reliable Tool Calling
- Prompt Chaining, Orchestration, Routing
- Building State Machines with LangGraph

## Tools / Stack
- ReAct
- Multi-Agent Orchestration
- LangGraph

## Weekly Build
**The Persistent Operator** — An autonomous, tool-calling agent that takes real actions and remembers customer context via Redis-backed persistent memory.

**Outcome:** Ship a tool-calling agent that pauses for human approval and resumes seamlessly.

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
