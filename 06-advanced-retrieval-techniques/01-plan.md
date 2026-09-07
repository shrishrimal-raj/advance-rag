# 🎯 Module 6 — Advanced Retrieval Techniques (Advanced Retriever)

> **Goal:** Graduate from "pick the top-k" to **smarter selection strategies** that fix the real
> weaknesses of basic retrieval: noisy chunks, missing context, unstructured questions, and
> ambiguous intent. This module builds a metadata-enriched corpus and runs **four** advanced
> retrievers on it.

---

## 🎯 Objectives

By the end of this module you will be able to:

1. Explain the **Contextual Compression Retriever** — retrieve more, let an LLM keep only the relevant sentences — and its token/latency trade-off.
2. Build a **Parent-Document Retriever** (the classic *small-to-big* pattern): index tiny children for precision, return large parents for context.
3. Implement a **Self-Query Retriever** that uses an LLM to turn natural language into a query **+ structured metadata filters** (needs a defined schema).
4. Implement a **Multi-Query Retriever** that rewrites an ambiguous question into N diverse queries, retrieves per query, then dedupes/merges.
5. Reason about **when to use each** and the **cost/latency** implications.

## 🧩 Prerequisites

- ✅ Module 5 (Basic Retrieval Techniques) — you're comfortable with retrievers, scores, and hybrid search.
- ✅ A **running LLM**: local Ollama (`ollama serve` + `ollama pull llama3.1`) or an `OPENAI_API_KEY`.
  - Retriever #2 (Parent-Document) is pure vector search and works without an LLM; #1, #3, #4 need one.
- ✅ `uv sync` completed; local embedding model available.

## 📦 Deliverables Checklist

- [ ] Read `02-learning.md` (theory + 2 architecture diagrams + cost table).
- [ ] Follow `03-implementation.md` step by step.
- [ ] Run `code/main.py`; confirm all four retrievers execute (LLM ones degrade gracefully if Ollama is down).
- [ ] Observe compression actually **shortening** the returned context.
- [ ] Observe parent-document returning a **larger** doc than the child that matched.
- [ ] Observe self-query adding a **metadata filter** (e.g. `format = csv`).
- [ ] Write your findings in `notes.md`.

## ⏱️ Time Estimate

| Activity | Time |
|----------|------|
| Reading theory + diagrams | 35 min |
| Setting up Ollama (if needed) | 10 min |
| Running & reading output | 25 min |
| Experiments (chunk sizes, num_queries) | 30 min |
| Notes | 10 min |
| **Total** | **~1.5–2 hours** |

## ✅ Success Criteria

You're done when you can answer, *without looking*:

- Why does compression save tokens but add latency?
- Why index small children but return big parents? What breaks if you only store parents?
- What must be true about your data for Self-Query filters to be useful?
- How does Multi-Query help an ambiguous question, and what's its cost?
- Which of the four would you reach for first in a customer-support bot, and why?
