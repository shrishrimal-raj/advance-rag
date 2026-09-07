# 🔨 Module 08 — Agentic RAG with LangGraph · Implementation

> This is the build guide for **both** implementations. Follow it top-to-bottom; each
> step maps to a block in the code files. Everything runs from **project root** with
> `uv run python 08-agentic-rag-langgraph/code/<script>.py`.

---

## Part 0 — Shared Setup (both scripts)

### Step 0.1 — Bootstrap the import path
Every script starts with the course boilerplate so it can import `shared.config` no matter
where you launch it from:

```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm, get_embeddings
```

`parents[2]` = project root (script lives in `NN-module/code/`). `get_llm()` returns an
OpenAI model if `OPENAI_API_KEY` is set, otherwise a local Ollama `llama3.1`.
`get_embeddings()` returns local MiniLM.

### Step 0.2 — Build a vector store over `data/samples/`
We read every file in `data/samples/` as a `Document`, chunk it, and load it into an
**in-memory** Chroma collection (no persist dir → always fresh, no cross-module state):

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma   # NOT langchain-chroma (not a dep)

def build_vector_store():
    docs = [Document(page_content=p.read_text(encoding="utf-8", errors="ignore"),
                     metadata={"source": p.name})
            for p in sorted(SAMPLES_DIR.iterdir()) if p.is_file()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = [c for d in docs for c in splitter.split_documents([d])]
    return Chroma.from_documents(chunks, get_embeddings(), collection_name="agentic_rag")
```

> Reading raw text (instead of per-format loaders) keeps the demo dependency-light and
> works for `.txt`, `.md`, `.json`, and `.csv` alike.

### Step 0.3 — A friendly LLM guard
Wrap LLM calls so a downed Ollama yields a hint, not a traceback:

```python
def _llm_call(llm, prompt, label):
    try:
        return llm.invoke([SystemMessage(content="You are a precise RAG pipeline component."),
                           HumanMessage(content=prompt)])
    except Exception as e:
        console.print(f"[red]{label} failed:[/red] {e}")
        console.print("[yellow]Is Ollama running? Try: ollama serve  (and: ollama pull llama3.1)[/yellow]")
        return None
```

---

## Part A — RAG as a Tool (`code/react_agent_rag.py`)

### Step A.1 — Define the tools
A tool is just a decorated function with a docstring (the docstring is what the LLM reads).

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for facts about RAG, vector databases/HNSW,
    Acme Robotics products, and the product catalog. Returns the top-3 most
    relevant passages with their source filenames."""
    hits = vs.similarity_search(query, k=3)          # <-- top-3 as required
    return "\n\n".join(f"[{i}] (source: {h.metadata.get('source','?')})\n{h.page_content.strip()}"
                       for i, h in enumerate(hits, 1)) or "No results found."

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as '1899.99 + 3*399.00'. Supports + - * / and parentheses."""
    ...  # safe eval (whitelist chars, then eval with empty builtins)
```

- `search_knowledge_base` is our **RAG-as-a-tool**: retrieval the agent may call *when it wants*.
- `calculator` is a second tool purely to demonstrate **multi-tool** use (the agent can
  search *and* do arithmetic).

### Step A.2 — Create the ReAct agent
```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(get_llm(temperature=0.0), tools=[search_knowledge_base, calculator])
```
`create_react_agent` builds the Thought→Action→Observation loop for us. The model must
support tool-calling (Ollama `llama3.1` does).

### Step A.3 — Run two questions and print the journey
We pass a system message + user question, then walk the returned `messages` list to show
each thought, tool call, tool result, and the final answer:

```python
for q in questions:
    result = agent.invoke({"messages": [("system", SYSTEM_PROMPT), ("user", q)]})
    _print_trace(result["messages"])
```

`_print_trace` maps message types to readable lines:
- `ai` **with** `tool_calls` → `🔧 tool_call: name(args)`
- `tool` → `↩ tool_result (name): <body>`
- `ai` **without** `tool_calls` → the **Final Answer** panel.

**Example questions** (chosen to force different tool paths):
1. *"What is the payload capacity of the Falcon AMR, and how many hours of battery does it have?"*
   → one `search_knowledge_base` call (company profile).
2. *"How much would it cost to buy one Quantum Laptop Pro 16 and three Orbit Ergonomic Chairs?"*
   → `search_knowledge_base` (products CSV) **then** `calculator("1899.99 + 3*399.00")`.

### Step A.4 — Safe calculator
Never `eval` raw model output. Whitelist characters first, then evaluate with empty builtins:
```python
if not re.fullmatch(r"[0-9+\-*/().\s]+", expr): raise ValueError(...)
return eval(expr, {"__builtins__": {}}, {})
```

---

## Part B — Explicit StateGraph (`code/agentic_graph_rag.py`)

This is the industry pattern: every stage is a node; the LLM's judgments become conditional
edges. We build it inside `main()` so nodes capture `vs` and `llm` via closure.

### Step B.1 — Typed State
```python
class State(TypedDict):
    question: str
    chat_history: Annotated[List[str], operator.add]   # reducer APPENDS across turns
    retrieved_docs: List[Document]
    graded_relevance: bool
    answer: str
    iteration: int                                       # safety counter (incremented in retrieve)
    reflection_supported: bool
    reflection_critique: str
    rewritten_query: Optional[str]
```
The only non-default field is `chat_history`, whose `operator.add` reducer makes it
accumulate — that's what gives the conversation memory.

### Step B.2 — The nodes (what each does + what it returns)

| Node | Reads | Does | Returns (state update) |
|------|-------|------|------------------------|
| `retrieve` | `question` / `rewritten_query`, `iteration` | Chroma `similarity_search(k=4)` | `retrieved_docs`, `iteration+1` |
| `grade_documents` | `retrieved_docs`, `question` | LLM grades each doc relevant/not; filter | `retrieved_docs` (filtered), `graded_relevance` |
| `rewrite_query` | `question` | LLM reformulates a sharper query | `rewritten_query` |
| `generate` | `retrieved_docs`, `question`, `chat_history` | LLM answers with `[n]` citations (or "not enough info" if no docs) | `answer`, appends to `chat_history` |
| `reflect` | `retrieved_docs`, `answer` | LLM checks answer is supported; may suggest refined query | `reflection_supported`, `reflection_critique`, optional `rewritten_query` |

Each node also calls a small `log(step, msg)` helper so the trace shows exactly what happened
inside it (which sources were pulled, how many docs survived grading, the reflection verdict).

**Grading & reflection use strict-JSON prompts** parsed by a tolerant `_parse_json`
(strips code fences, grabs the first `{...}`). If parsing fails we degrade safely
(keep all docs / assume unsupported) rather than crash.

### Step B.3 — Conditional edges (the "agentic" part)
Two routing functions turn LLM judgments into control flow:

```python
def route_after_grade(state):
    if state["graded_relevance"]:            return "generate"      # good evidence
    if state["iteration"] < MAX_ITERATIONS:  return "rewrite_query" # retry with new query
    return "generate"                          # give up -> honest 'not enough info'

def route_after_reflect(state):
    if state["reflection_supported"]:         return END            # grounded answer
    if state["iteration"] < MAX_ITERATIONS:  return "retrieve"      # loop back & re-retrieve
    return END                                   # exhausted -> stop
```

### Step B.4 — Wire the graph
```python
g = StateGraph(State)
g.add_node("retrieve", retrieve); g.add_node("grade_documents", grade_documents)
g.add_node("rewrite_query", rewrite_query); g.add_node("generate", generate)
g.add_node("reflect", reflect)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "grade_documents")
g.add_conditional_edges("grade_documents", route_after_grade, ["generate", "rewrite_query"])
g.add_edge("rewrite_query", "retrieve")
g.add_edge("generate", "reflect")
g.add_conditional_edges("reflect", route_after_reflect, ["retrieve", END])

app = g.compile(checkpointer=MemorySaver())
```

Edge-by-edge:
- `START → retrieve`: always start by retrieving.
- `retrieve → grade_documents`: always grade what we retrieved.
- `grade_documents → {generate | rewrite_query}`: branch on relevance.
- `rewrite_query → retrieve`: close the *relevance* feedback loop.
- `generate → reflect`: always self-check the answer.
- `reflect → {retrieve | END}`: close the *grounding* feedback loop or finish.

### Step B.5 — Run a 2-turn conversation (checkpointing/memory)
Use one `thread_id`; the `MemorySaver` persists state between turns, and the
`chat_history` reducer accumulates it. Each turn we reset the volatile fields
(`iteration`, `rewritten_query`, …) but keep history:

```python
config = {"configurable": {"thread_id": "demo-conversation"}}
for i, q in enumerate(turns, 1):
    initial = {"question": q, "iteration": 0, "rewritten_query": None,
               "graded_relevance": False, "reflection_supported": False,
               "answer": "", "chat_history": [f"User: {q}"]}
    app.invoke(initial, config)                 # nodes print their own trace
    final = app.get_state(config).values
    console.print(Panel.fit(final["answer"], title=f"Turn {i} · Answer"))
    console.print(f"[dim]Memory: {final['chat_history']}[/dim]")
```

Because `generate` includes `chat_history` in its prompt, **Turn 2 can reference Turn 1**
(e.g. "Now, which Acme product has the larger payload…").

### Step B.6 — Trace of node transitions
Two complementary signals make the journey visible:
1. **Inside each node**, `log(...)` prints what it did (sources pulled, docs kept, verdict).
2. **Routing functions** print the branch taken ("no relevant docs → rewrite_query").
3. A per-turn **journey list** (each node appends its name) is printed at the end, e.g.
   `retrieve → grade_documents → rewrite_query → retrieve → grade_documents → generate → reflect`.

---

## Part C — Run It

```bash
# from project root
uv run python 08-agentic-rag-langgraph/code/react_agent_rag.py
uv run python 08-agentic-rag-langgraph/code/agentic_graph_rag.py
```

First run downloads the MiniLM embedding model (~90 MB) — subsequent runs are fast.

---

## Part D — Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Connection refused` / friendly Ollama hint | Ollama not running | `ollama serve`, then `ollama pull llama3.1` |
| Agent never calls the search tool | Model too small / weak tool-calling | Use `llama3.1` (or larger); strengthen the system prompt |
| Grading/reflection JSON parse fails | Model wrapped JSON in prose | `_parse_json` already tolerates this; else bump the model |
| Graph loops forever | Missing iteration cap | Ensure `MAX_ITERATIONS` guards both routing functions |
| Slow first run | Embedding model download | One-time; check network / proxy |
| `ModuleNotFoundError: shared` | Ran from wrong cwd / missing boilerplate | Keep the `sys.path.append(parents[2])` line at the top |
