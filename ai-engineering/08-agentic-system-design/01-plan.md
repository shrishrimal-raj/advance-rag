# 🎯 Week 08 — Agentic System Design

> Status: ✅ BUILT (see `../CHECKPOINT.md`).

## Objective
Design robust, production-grade agentic systems.

## What You'll Learn
- Agentic System Design Patterns
- Tool Use & Function Calling
- Planning & Memory Architectures
- Error Handling & Recovery Strategies

## Tools / Stack
- **LangGraph** (installed) · **LangChain** (installed) · **shared.config.get_llm** (cloud)

## Weekly Build
**The Autonomous Research Agent** — A multi-step agent that plans, executes, and self-corrects.

**Outcome:** Master the architecture behind reliable, production-grade agents.

## Approach details
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | LangGraph autonomous agent: decide -> act -> observe loop with self-correction | cloud LLM (`--selftest` ≈ 2–3 calls, capped) |
| `code/approach_2_raw_function_calling.py` | Hand-rolled function-calling round-trip (schema -> call -> dispatch -> result) | **fully offline** |
| `code/approach_3_planning_memory_from_scratch.py` | Planner + memory store from scratch | **fully offline** |
| `code/benchmark_agent_design_patterns.py` | ReAct vs plan-and-execute vs reflection decision matrix | **fully offline** |

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` · `approach_2_raw_function_calling.py` · `approach_3_planning_memory_from_scratch.py` · `benchmark_agent_design_patterns.py`
- [x] All scripts: UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four.
- Run `approach_2` + `approach_3` + `benchmark` (offline, self-check asserts).
- Run `main.py --selftest` (bounded cloud LLM loop).

## Prerequisites
- Weeks 1–7 complete. `langgraph` installed. No Ollama; no new installs.

## Time Estimate
~5–7 hours hands-on.
