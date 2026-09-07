# 🎯 Week 04 — AI Agents & State Machines

> Status: ✅ BUILT (see `../CHECKPOINT.md`).

## Objective
Build autonomous AI systems that reason, use tools, and manage complex workflows.

## What You'll Learn
- LangGraph StateGraph Architecture
- ReAct Pattern Implementation
- Tool Calling & Function Execution
- Human-in-the-Loop Workflows

## Tools / Stack
- **LangGraph** (StateGraph / prebuilt ReAct) · **Pydantic** (typed state/tools) · **shared.config.get_llm** (cloud)

## Weekly Build
**The Customer Support Agent** — An autonomous agent that searches docs, looks up orders, and escalates to humans.

**Outcome:** Master agentic patterns used in production AI products.

## Approach details
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | LangGraph prebuilt ReAct agent with 3 tools | cloud LLM (`--selftest` = 1 call) |
| `code/approach_2_raw_react_loop.py` | Hand-rolled ReAct/tool-call loop, no framework | cloud LLM (`--selftest` = 1 call) |
| `code/approach_3_state_machine_hitl.py` | Rule-based FSM with a human-approval gate | **fully offline** |
| `code/benchmark_agent_vs_direct_rag.py` | What an agent can reach vs plain RAG | **fully offline** |

> Only `main.py` and `approach_2` touch the LLM (one call each under `--selftest`). The FSM and the benchmark are deterministic/offline.

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` · `approach_2_raw_react_loop.py` · `approach_3_state_machine_hitl.py` · `benchmark_agent_vs_direct_rag.py`
- [x] All scripts: UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four.
- Run `approach_3` + `benchmark_agent_vs_direct_rag` (offline, self-check asserts).
- Run `main.py --selftest` and `approach_2_raw_react_loop.py --selftest` (1 cloud LLM call each).

## Prerequisites
- Weeks 1–3 complete. `langgraph` installed; Yolo-Auto key set. No Ollama.

## Time Estimate
~5–7 hours hands-on.
