# 🎯 Week 06 — MCP, Context & Multi-Agent

> Status: ✅ BUILT (see `../CHECKPOINT.md`).

## Objective
Build systems that connect to external tools via standard protocols and manage complex workflows.

## What You'll Learn
- Model Context Protocol (MCP)
- Context Engineering & Window Management
- Multi-Agent Orchestration Patterns

## Tools / Stack
- **LangGraph** (StateGraph multi-agent) · **MCP** (protocol, hand-rolled here; `mcp`/`fastmcp` SDKs documented) · **shared.config.get_llm** (cloud)

> Note: the Python `mcp`/`fastmcp` SDKs are not preinstalled. To teach the protocol *and* stay runnable anywhere, `approach_2` **hand-rolls the MCP JSON-RPC handshake** (initialize -> tools/list -> tools/call) in-process. `main.py` uses LangGraph multi-agent with local tools and treats MCP as an *optional* tool source. The real SDKs are covered in `02-learning.md`.

## Weekly Build
**The Research Assistant** — An MCP-powered multi-agent research system that plans, gathers, and synthesizes.

**Outcome:** Master the protocols & patterns behind modern agentic products.

## Approach details
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | LangGraph StateGraph multi-agent (plan -> research -> synthesize) | cloud LLM (`--selftest` = 2 calls) |
| `code/approach_2_raw_mcp_client.py` | Hand-rolled MCP JSON-RPC client+server (initialize/list/call) | **fully offline** |
| `code/approach_3_context_management_from_scratch.py` | Token-budget context manager (keep system+latest, evict/summarize) | **fully offline** |
| `code/benchmark_single_vs_multi_agent.py` | Context-load: single vs multi-agent | **fully offline** |

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` · `approach_2_raw_mcp_client.py` · `approach_3_context_management_from_scratch.py` · `benchmark_single_vs_multi_agent.py`
- [x] All scripts: UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four.
- Run `approach_2` + `approach_3` + `benchmark` (all offline, self-check asserts).
- Run `main.py --selftest` (2 cloud LLM calls).

## Prerequisites
- Weeks 1–5 complete. `langgraph` installed. No Ollama; no new installs needed.

## Time Estimate
~5–7 hours hands-on.
