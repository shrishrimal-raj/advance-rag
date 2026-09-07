# 🔧 Week 4 Implementation — Build The Customer Support Agent

> Step-by-step. Each step ends with a **verify** line.

## 0. Setup
```bash
# from repo root (the course venv lives at ./.venv)
./.venv/bin/python -c "import langgraph, pydantic; print('ok')"
```
**verify:** prints `ok`.

## 1. Define the tools
Three `@tool` functions with clear docstrings:
- `search_docs(query)` — keyword search over the KB.
- `lookup_order(order_id)` — fetch from a fake order DB.
- `escalate_to_human(reason)` — record an escalation.
**verify:** each tool returns a string when called directly.

## 2. Wire the LangGraph agent
`create_react_agent(get_llm(), [tools])`. Invoke with a user message; print the final message.
**verify:** `python code/main.py --selftest` runs one query end-to-end (cloud LLM).

## 3. The raw ReAct loop (approach_2)
Bind tools to the LLM, then loop: invoke -> if `tool_calls`, execute each and append `ToolMessage`s; else return content. Cap iterations.
**verify:** `python code/approach_2_raw_react_loop.py --selftest` completes in <= N steps.

## 4. The FSM with HITL (approach_3)
A dict-driven state machine: `COLLECTING -> VERIFYING -> AWAITING_APPROVAL -> ESCALATED|DONE`. Simulate a human decision at the gate. No LLM.
**verify:** `python code/approach_3_state_machine_hitl.py` prints the state trace and passes its self-check.

## 5. Agent vs RAG benchmark
Show which queries plain RAG can answer (corpus only) vs which need the agent's live order data. Offline.
**verify:** `python code/benchmark_agent_vs_direct_rag.py` shows agent >= RAG and passes its self-check.

## 6. Run everything
```bash
PY=./.venv/bin/python   # or .venv\Scripts\python.exe
$PY code/approach_3_state_machine_hitl.py
$PY code/benchmark_agent_vs_direct_rag.py
$PY code/main.py --selftest
$PY code/approach_2_raw_react_loop.py --selftest
```

## Troubleshooting
- **Agent loops forever** — cap max steps; ensure a tool can always lead to a final answer.
- **Tool args wrong type** — tighten the docstring/schema; the model follows types.
- **No final answer** — confirm the model stops emitting tool_calls when it has enough info.
- **HITL gate stuck** — the FSM must always transition out of AWAITING_APPROVAL on a decision.
