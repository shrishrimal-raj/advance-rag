# ✍️ Module 08 — Agentic RAG with LangGraph · My Notes

> Fill this in as you go. Short, honest notes beat long summaries.

## 🧠 One-line takeaway
_What is the single biggest idea from this module?_
- 

## 🔁 Agentic vs. fixed pipeline
- Why hand control to the LLM?
  - 
- What breaks when I do? (cost, loops, unpredictability)
  - 

## 🛠️ Approach A — ReAct agent (RAG as a tool)
- What tools did I expose?
  - 
- Did the agent pick the right tool for each question? Show the trace:
  - Q1: 
  - Q2: 
- When would I reach for `create_react_agent`?
  - 

## 🕸️ Approach B — explicit StateGraph
- List my nodes and what each returns:
  - `retrieve`: 
  - `grade_documents`: 
  - `rewrite_query`: 
  - `generate`: 
  - `reflect`: 
- My two conditional edges (what triggers each branch):
  - after `grade_documents`: 
  - after `reflect`: 
- How does the graph guarantee it stops?
  - 

## 🧩 A vs B — my decision rule
- Pick **A** when…
  - 
- Pick **B** when…
  - 

## 🧠 Memory / checkpointing
- What did Turn 2 remember from Turn 1?
  - 
- What would change if I used a real checkpointer (SQLite/Postgres) instead of `MemorySaver`?
  - 

## ⚠️ Gotchas I hit
- 
- 

## 🚀 Production ideas (for Module 11)
- Max iterations / recursion limit: 
- Cost control: 
- Observability: 
- Guardrails: 

## ❓ Questions for later
- 
- 

## ✅ Self-check (tick when true)
- [ ] I can explain ReAct in one sentence.
- [ ] I can explain why conditional edges make RAG "agentic."
- [ ] Both scripts ran locally with no API keys.
- [ ] I saw a re-retrieve loop fire in Approach B.
- [ ] I know when to use A vs B.
