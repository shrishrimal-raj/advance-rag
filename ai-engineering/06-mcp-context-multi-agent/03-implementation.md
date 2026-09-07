# 🔧 Week 6 Implementation — Build The Research Assistant

> Step-by-step. Each step ends with a **verify** line.

## 0. Setup
```bash
./.venv/bin/python -c "import langgraph; print('ok')"
```
**verify:** prints `ok`.

## 1. Hand-roll the MCP protocol (approach_2)
A minimal in-process JSON-RPC server exposing one tool (`lookup`) and a client that performs `initialize`, `notifications/initialized`, `tools/list`, `tools/call`. Assert the round-trip.
**verify:** `python code/approach_2_raw_mcp_client.py` prints the message trace and passes its self-check.

## 2. Context manager (approach_3)
`fit_context(system, messages, budget)`: always keep system + latest; fill remaining budget newest-first; summarize evicted turns into a running note. Token proxy = word count.
**verify:** `python code/approach_3_context_management_from_scratch.py` fits the budget and passes its self-check.

## 3. Single vs multi-agent (benchmark)
Compute per-approach context load for a task with K sub-skills. Show multi-agent keeps each node's context smaller than the single-agent monolith.
**verify:** `python code/benchmark_single_vs_multi_agent.py` shows multi < single and passes its self-check.

## 4. The multi-agent graph (main.py)
LangGraph `StateGraph`: `plan` (LLM -> sub-questions) -> `research` (deterministic tool calls over a local corpus; MCP tools added if available) -> `synthesize` (LLM -> cited answer). Shared state carries plan/findings/answer.
**verify:** `python code/main.py --selftest` runs one query end-to-end (2 LLM calls).

## 5. Run everything
```bash
PY=./.venv/bin/python   # or .venv\Scripts\python.exe
$PY code/approach_2_raw_mcp_client.py
$PY code/approach_3_context_management_from_scratch.py
$PY code/benchmark_single_vs_multi_agent.py
$PY code/main.py --selftest
```

## Troubleshooting
- **MCP id mismatch** — every response `id` must echo its request `id`; notifications have no `id`.
- **Context still over budget** — shrink the summary or drop more history; never drop the system prompt.
- **Agent loop** — cap iterations; ensure synthesize always terminates.
- **No MCP server** — expected here; local tools are the fallback, MCP is opt-in.
