# 🔨 Module 10 — Implementation: Building Papeer Phase by Phase

All commands run from the **project root** (`D:\Raj\Projects\advance-rag`) with `uv`.
Every code file lives in `10-capstone-project/code/papeer/` and is runnable via
`uv run python 10-capstone-project/code/papeer/<file>.py` (except `api.py`, which needs extra deps).

> **Prereq:** `uv sync` done, and (for zero-key runs) Ollama running with `llama3.1`:
> `ollama serve` then `ollama pull llama3.1`. Embeddings auto-download on first run.

---

## Phase 1 — Ingestion (`ingestion.py`)

**What it does:** loads every supported file in `data/samples/` (plus an optional user dir),
chunks with metadata, embeds, persists to a persistent Chroma collection, and writes a JSON
sidecar (`chunks.json`) used by BM25.

**Key ideas to notice while reading:**
- Format-aware loaders → normalized LangChain `Document`s.
- Stable chunk IDs = hash of `(source, chunk_index)` → idempotent upserts.
- Metadata: `source`, `page`, `chunk_index`, `doc_type`.

**Run it:**
```bash
uv run python 10-capstone-project/code/papeer/ingestion.py
```
**Expected:** prints per-file chunk counts and total; creates `data/chroma_db/papeer/` and
`data/chroma_db/papeer_chunks.json`.

---

## Phase 2 — Hybrid Retrieval + Rerank (`retrieval.py`)

**What it does:** given a question, runs dense Chroma search (top-10) and BM25 over the sidecar
(top-10), fuses with Reciprocal Rank Fusion, then cross-encoder reranks the fused set to top-3.

**Key ideas:**
- RRF formula `Σ 1/(k + rank)`, `k=60` — no score normalization needed.
- Cross-encoder only on the small fused set (two-stage retrieve-then-rerank).
- Returns passages with `text`, `source`, `score` for citations.

**Run it (self-test):**
```bash
uv run python 10-capstone-project/code/papeer/retrieval.py "What navigation does the Falcon AMR use?"
```
**Expected:** prints the top-3 ranked passages with source + score.

---

## Phase 3 — Agentic Workflow (`agent.py`)

**What it does:** builds a LangGraph `StateGraph` with nodes `retrieve → grade → generate →
reflect`, bounded to max 2 iterations. Records per-stage wall-clock timings in state.

**Key ideas:**
- State is a `TypedDict`; conditional edges decide whether to re-retrieve or finish.
- `grade` gates generation; `reflect` decides whether to loop.
- `run_agent(question)` returns `{answer, context, stages:[(name, secs), ...]}`.

**Run it (self-test):**
```bash
uv run python 10-capstone-project/code/papeer/agent.py "Summarize Acme Robotics' support policy."
```
**Expected:** prints the answer and a small table of stage timings.

---

## Phase 4 — Evaluation (`evaluate.py`)

**What it does:** defines a small built-in golden set, runs each question through retrieval +
generation, then scores with RAGAS (faithfulness, answer relevancy, context precision/recall)
and prints a report. Degrades gracefully if RAGAS isn't importable.

**Run it:**
```bash
uv run python 10-capstone-project/code/papeer/evaluate.py
```
**Expected:** a rich table of metrics; faithfulness target > 0.8.

---

## Phase 5 — Deployment & Demo

### CLI demo (`main.py`) — runnable
Runs ingestion once, then asks **3 questions** through the full pipeline, printing stages,
answers, and per-stage timing.
```bash
uv run python 10-capstone-project/code/papeer/main.py
```

### API service (`api.py`) — NOT runnable without extra deps
Exposes `POST /ask` with a streaming-ready structure. **Before running:**
```bash
uv add fastapi uvicorn
uv run uvicorn 10-capstone-project.code.papeer.api:app --reload --port 8000
# then: POST http://localhost:8000/ask  {"question": "..."}
```
(See the note at the top of `api.py`.)

---

## Build order & verification

| Step | Command | Verify |
|------|---------|--------|
| 1 | `.../papeer/ingestion.py` | index + sidecar created, chunk count > 0 |
| 2 | `.../papeer/retrieval.py "<q>"` | top-3 passages printed with sources |
| 3 | `.../papeer/agent.py "<q>"` | answer + stage timings |
| 4 | `.../papeer/evaluate.py` | metrics table, faithfulness > 0.8 |
| 5 | `.../papeer/main.py` | 3 questions, per-stage timing |
| 6 | `api.py` (after `uv add fastapi uvicorn`) | `POST /ask` returns answer |

## Troubleshooting
- **No Ollama / connection refused:** start `ollama serve`, `ollama pull llama3.1`, or set
  `OPENAI_API_KEY` in `.env` to use cloud LLMs.
- **Embedding download slow:** first run fetches MiniLM (~90 MB); subsequent runs are cached.
- **Stale index after changing chunk size or embedding model:** delete `data/chroma_db/papeer*`
  and re-run ingestion (vectors are not portable across models).
- **RAGAS import error:** `evaluate.py` falls back to a manual report; install/upgrade ragas to
  get full metrics.
