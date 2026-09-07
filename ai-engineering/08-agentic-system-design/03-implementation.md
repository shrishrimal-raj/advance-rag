# 🔧 Week 8 Implementation — Build The Autonomous Research Agent

> Step-by-step. Each step ends with a **verify** line.

## 0. Setup
```bash
./.venv/bin/python -c "import langgraph; print('ok')"
```
**verify:** prints `ok`.

## 1. Function-calling contract (approach_2)
Define tools with JSON schemas; a stand-in "model" emits a tool call; the harness parses, dispatches to the real function, and feeds the result back. Assert the round-trip and that a bad arg is rejected.
**verify:** `python code/approach_2_raw_function_calling.py` completes the round-trip and passes its self-check.

## 2. Planning + memory (approach_3)
A rule-based planner decomposes a goal into subtasks, executes them, stores results in a memory dict, and retrieves relevant memory for a follow-up. Assert decomposition, execution, and recall.
**verify:** `python code/approach_3_planning_memory_from_scratch.py` plans, executes, and recalls correctly.

## 3. Pattern comparison (benchmark)
Score ReAct vs plan-and-execute vs reflection across cost/reliability/latency/adaptability for two scenarios; recommend. Assert the right pattern wins each scenario.
**verify:** `python code/benchmark_agent_design_patterns.py` recommends correctly and passes its self-check.

## 4. The autonomous agent (main.py)
LangGraph loop: `decide` (LLM -> tool call OR final) -> `act` (run search/fetch over a local corpus) -> observe -> back to `decide`. Guards: max iterations, tool-input validation, graceful stop. Self-correction: the model can re-query with better args.
**verify:** `python code/main.py --selftest` runs one query through the bounded loop (≈2–3 LLM calls).

## 5. Run everything
```bash
PY=./.venv/bin/python   # or .venv\Scripts\python.exe
$PY code/approach_2_raw_function_calling.py
$PY code/approach_3_planning_memory_from_scratch.py
$PY code/benchmark_agent_design_patterns.py
$PY code/main.py --selftest
```

## Troubleshooting
- **Runaway loop** - enforce the iteration cap; always have a `final` path.
- **Bad tool args** - validate against the schema; on failure, send a corrective observation and retry once.
- **Hallucinated id** - `fetch` must handle unknown ids gracefully (return a not-found observation, not crash).
- **Too many LLM calls** - prefer plan-and-execute for known procedures; reserve ReAct for open-ended tasks.
