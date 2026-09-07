# 🎯 Week 05 — Evals & Observability

> Status: ✅ BUILT (see `../CHECKPOINT.md`).

## Objective
Measure, monitor, and guard the quality of AI systems.

## What You'll Learn
- RAGAS / DeepEval Metrics (faithfulness, relevancy, precision, recall)
- Langfuse Tracing & Observability
- Automated Regression Testing for AI

## Tools / Stack
- **RAGAS** (installed - real metrics, opt-in/LLM) · **Langfuse** (optional tracer; JSONL fallback) · **pytest** · custom numpy metrics (offline core)

> Note: `deepeval`/`langfuse` are not preinstalled here. The runnable core uses **RAGAS** (present) + **custom offline metrics**; Langfuse is wired as an *optional* tracer with a JSONL fallback so the gate runs anywhere. DeepEval is covered as the alternative framework in `02-learning.md`.

## Weekly Build
**The AI Regression Gate** — An automated eval pipeline with tracing + regression detection that blocks bad changes.

**Outcome:** Master the evaluation & monitoring practices used in production AI.

## Approach details
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | Regression gate: offline metrics + tracing (langfuse-or-JSONL) + baseline diff | **fully offline** |
| `code/approach_2_ragas.py` | Real RAGAS metrics (`ragas.metrics.collections`) | `--dry-run` offline; `--run` needs LLM |
| `code/approach_3_custom_metrics_from_scratch.py` | faithfulness/recall/correctness math, pure python | **fully offline** |
| `code/benchmark_regression_detector.py` | baseline-vs-current regression detection | **fully offline** |

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` · `approach_2_ragas.py` · `approach_3_custom_metrics_from_scratch.py` · `benchmark_regression_detector.py`
- [x] All scripts: UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four.
- Run `approach_3` + `benchmark_regression_detector` + `main.py` (all offline, self-check asserts).
- Run `approach_2_ragas.py --dry-run` (offline; confirms imports + sample construction).

## Prerequisites
- Weeks 1–4 complete. `ragas` installed. No Ollama; no new installs needed.

## Time Estimate
~4–6 hours hands-on.
