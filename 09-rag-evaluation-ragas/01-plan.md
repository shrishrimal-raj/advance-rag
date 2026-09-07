# 🎯 Module 09 — RAG Evaluation with RAGAS

> **One-liner:** A RAG system you can't measure is a RAG system you can't ship. This module turns "it feels better" into numbers, so you can prove your pipeline works, catch regressions, and tune with confidence.

---

## Objectives

By the end of this module you will be able to:

1. **Explain why evaluation is the single biggest gap between a toy RAG demo and a production RAG system.**
2. **Separate retrieval quality from generation quality** — and know *which* metric belongs to *which* stage of the pipeline.
3. **Deeply understand the four core RAGAS metrics** (Faithfulness, Answer Relevancy, Context Precision, Context Recall): what each measures, its formula/intuition, and concrete failure examples.
4. **Build a golden test dataset** (question + ground_truth + contexts) from real documents.
5. **Wire a live retriever + generator into RAGAS** and run both individual metric scores and a full `ragas.evaluate()` run.
6. **Interpret results** and set **baselines + regression thresholds** for CI.
7. **Reason about LLM-as-judge caveats** (bias, cost, non-determinism) so you don't get fooled by your own evaluator.

---

## Prerequisites

- Completed earlier modules: document loading/chunking, vector store setup (Chroma), and a working RAG chain (retriever → generator).
- `uv` installed and `uv sync` run in the project root (installs `ragas>=0.2`, `datasets`, `langchain 0.3`, `chromadb`, `sentence-transformers`).
- **No API keys required.** Local Ollama LLM (`llama3.1`) + local `all-MiniLM-L6-v2` embeddings are used by default.
  - If using Ollama: `ollama serve` running and `ollama pull llama3.1` done.
  - Optional speed-up: set `OPENAI_API_KEY` in `.env` to use OpenAI as the judge (much faster than a local LLM).
- Sample documents present in `data/samples/` (company_profile.json, products.csv, rag_overview.txt, vector_db_notes.md).

---

## Deliverables checklist

| # | Deliverable | File | Status |
|---|-------------|------|--------|
| 1 | Planning doc (this file) | `01-plan.md` | ✅ |
| 2 | Learning doc w/ 2+ mermaid diagrams & metric deep-dives | `02-learning.md` | ⬜ |
| 3 | Step-by-step implementation guide | `03-implementation.md` | ⬜ |
| 4 | Learner notes template | `notes.md` | ⬜ |
| 5 | Golden dataset builder (8–10 examples) | `code/build_dataset.py` | ⬜ |
| 6 | Pipeline evaluator (individual + full RAGAS) | `code/evaluate_pipeline.py` | ⬜ |
| 7 | Generated dataset artifact | `data/eval_dataset.json` (created by #5) | ⬜ |

**Run order:**
```bash
uv run python 09-rag-evaluation-ragas/code/build_dataset.py      # builds data/eval_dataset.json
uv run python 09-rag-evaluation-ragas/code/evaluate_pipeline.py  # scores the pipeline
```

---

## Time estimate (~3 hours)

| Block | Activity | Time |
|-------|----------|------|
| 1 | Read `02-learning.md`; internalize the 4 core metrics + failure modes | 45 min |
| 2 | Run `build_dataset.py`; inspect the golden dataset; hand-tune 1–2 ground truths | 25 min |
| 3 | Run `evaluate_pipeline.py` (individual metrics on 2 examples) | 30 min |
| 4 | Run full `ragas.evaluate()`; read the scorecard | 30 min |
| 5 | Experiment: break the pipeline on purpose (bad prompt / wrong k / drop a doc) and watch scores move | 30 min |
| 6 | Write baselines + CI thresholds in `notes.md`; reflect on judge caveats | 20 min |
| **Total** | | **~3 h** |

> ⏱️ **Heads-up:** RAGAS uses an LLM as a judge. On a local Ollama model each metric can take many seconds per example, so a full run over 10 examples × 4 metrics is slow (tens of minutes). Use `EVAL_LIMIT=3` to cap the full run while learning, or switch to OpenAI in `.env` for speed.

---

## Success criteria

You're done when you can answer **"yes"** to all of these:

- [ ] I can state, for each of the 4 core metrics, **which pipeline stage it evaluates** and **what a low score means**.
- [ ] I have a `data/eval_dataset.json` with **8–10 examples**, each having a question, a hand-written ground truth, and the correct source context(s).
- [ ] I ran individual metric scores on 2 examples and can explain **why** each number is high or low.
- [ ] I ran a full `ragas.evaluate()` and produced a **summary table** of mean scores.
- [ ] I deliberately degraded the pipeline once and **watched the relevant metric drop** (proving the metric actually measures what I think it does).
- [ ] I recorded a **baseline** (e.g., Faithfulness ≥ 0.8, Context Recall ≥ 0.7) and a **regression threshold** I'd enforce in CI.
- [ ] I can name at least **two LLM-as-judge failure modes** (e.g., self-preference bias, verbosity bias) and one mitigation.
