# 🎯 Module 08 — Agentic RAG with LangGraph · Plan

> **Theme:** Stop hard-coding the RAG pipeline. Let the LLM *decide* when to
> retrieve, whether the retrieved docs are good enough, and whether its own
> answer is trustworthy — using **LangGraph**.

---

## 🧭 Overview

Modules 01–07 built a **fixed** RAG pipeline: `Question → Retrieve → Generate`.
Every question takes the exact same path, no matter how easy or hard it is.

In this module we make RAG **agentic**: the LLM becomes the *controller*. It
chooses **when** to search, **how** to reformulate a bad search, **whether** the
retrieved evidence is relevant, and **whether** its drafted answer is actually
supported by that evidence. We build this two ways:

| Approach | File | Idea |
|----------|------|------|
| **A — RAG as a Tool** | `code/react_agent_rag.py` | Expose retrieval as a *tool*; a prebuilt **ReAct** agent calls it (plus a `calculator`) as it reasons. Fast to build, flexible. |
| **B — RAG as Graph Nodes** | `code/agentic_graph_rag.py` | Hand-build a **StateGraph** where each RAG stage is a *node* and the LLM drives *conditional edges* (grade → re-retrieve, reflect → retry). More control, the industry pattern. |

Both run **locally with zero API keys** (Ollama `llama3.1` + local
`sentence-transformers` embeddings) and print a clean trace of the agent's journey.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

1. **Explain** what makes RAG "agentic" — the LLM controls the *flow*, not just the words.
2. **Implement** a ReAct agent with `langgraph.prebuilt.create_react_agent` that uses
   retrieval **as a tool**, alongside a second tool (multi-tool use).
3. **Build** an explicit `StateGraph` with typed state and named nodes
   (`retrieve`, `grade_documents`, `rewrite_query`, `generate`, `reflect`).
4. **Wire** **conditional edges** so the graph branches on LLM judgments
   (relevant vs. not, supported vs. unsupported) and can **loop** back to re-retrieve.
5. **Add memory** with a `MemorySaver` checkpointer and run a multi-turn conversation.
6. **Apply production safeguards**: max-iteration caps, graceful failure, and cost control.

---

## 📚 Prerequisites

**From earlier modules**
- Module 04 (Vector Stores) — you know how to load a corpus into Chroma.
- Module 05/06 (Retrieval) — you understand similarity search and why relevance matters.
- Module 07 (Advanced Patterns) — you've seen CRAG / Self-RAG ideas; this module *implements* them as agents.

**LangGraph basics (the core of this module)**
- **`StateGraph`** — a directed graph whose nodes mutate a shared **State**.
- **Nodes** — plain functions `(state) -> partial_state_update`.
- **Edges** — normal edges (`A → B`) and **conditional edges** (a function picks the next node).
- **State** — a `TypedDict`; fields can have **reducers** (e.g. `operator.add` to append).
- **Checkpointing / memory** — a `checkpointer` (e.g. `MemorySaver`) persists state per `thread_id`, enabling multi-turn conversations and resumption.
- **Human-in-the-loop** — interrupt/resume (we mention it; our demos are autonomous).

**Python**
- Closures, `typing.TypedDict`, `Annotated`, `try/except`.

**Local environment**
- `uv` installed and `uv sync` run from project root.
- [Ollama](https://ollama.com) running: `ollama serve` and `ollama pull llama3.1`.
- Local embeddings auto-download on first run (`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB).

---

## 🧰 Tools & Libraries Used

| Library | Role |
|---------|------|
| `langgraph` ≥ 0.2 | `StateGraph`, `create_react_agent`, `MemorySaver`, `START`/`END` |
| `langchain` 0.3 | `@tool`, messages, `get_llm` factory |
| `langchain-community` | `Chroma` vector-store wrapper (over `chromadb`) |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` for chunking |
| `sentence-transformers` | Local embeddings (MiniLM) |
| `rich` | Pretty console output (panels, rules, dim traces) |

> **Note:** `langchain-chroma` is *not* a dependency here, so we use
> `langchain_community.vectorstores.Chroma` (already available) — same behaviour.

---

## 📦 Deliverables Checklist

- [ ] `01-plan.md` — this file
- [ ] `02-learning.md` — theory + **≥ 2 mermaid diagrams** with explanations
- [ ] `03-implementation.md` — step-by-step build guide for **both** implementations
- [ ] `code/react_agent_rag.py` — ReAct agent, `search_knowledge_base` + `calculator` tools,
      runs **2 example questions**, prints reasoning steps / tool calls / results
- [ ] `code/agentic_graph_rag.py` — explicit `StateGraph` (retrieve → grade → generate → reflect,
      with `rewrite_query` + loops), `MemorySaver` checkpointer, prints **every node transition**
- [ ] `notes.md` — your takeaways (template provided)

**Definition of done:** both scripts run from project root with
`uv run python 08-agentic-rag-langgraph/code/<script>.py`, need **no API keys**,
degrade gracefully (friendly Ollama hint) if the server is down, and clearly show
the agent making decisions.

---

## ⏱️ Time Estimate (~4 hours)

| Block | Activity | Time |
|-------|----------|------|
| 1 | Read `02-learning.md`, study the two diagrams | 45 min |
| 2 | Set up local env (Ollama + embeddings smoke test) | 15 min |
| 3 | Build **A** — ReAct agent with RAG tool; run 2 questions | 60 min |
| 4 | Build **B** — explicit StateGraph with conditional edges + loops | 90 min |
| 5 | Add `MemorySaver` multi-turn conversation + trace polish | 30 min |
| 6 | Break it on purpose (kill Ollama, force a bad retrieval) & fix | 20 min |
| 7 | Write `notes.md`, compare A vs B | 20 min |
| | **Total** | **~4 h** |

---

## ✅ Success Criteria

You're done when **all** of these are true:

1. `uv run python 08-agentic-rag-langgraph/code/react_agent_rag.py` runs and you can
   *see* the agent call `search_knowledge_base` **and** `calculator` before answering.
2. `uv run python 08-agentic-rag-langgraph/code/agentic_graph_rag.py` runs and the trace
   shows the sequence `retrieve → grade_documents → generate → reflect`, **and** at least
   one conditional branch fires (a re-retrieve loop or a "not enough info" path).
3. Killing Ollama produces a **friendly hint** (not a stack trace) in both scripts.
4. The graph script runs a **2-turn conversation** and the second turn remembers the first
   (via the checkpointer).
5. You can verbally explain, in one sentence each, *when* you'd pick Approach A vs Approach B.

---

## 🔗 Where This Fits

- **Before:** Module 07 introduced CRAG / Self-RAG *concepts*. Here we **implement** them as
  controllable graphs.
- **After:** Module 09 (RAGAS evaluation) will score the answers these agents produce;
  Module 11 (Production RAG) will add caching, monitoring, and guardrails around exactly
  this kind of agentic flow.
