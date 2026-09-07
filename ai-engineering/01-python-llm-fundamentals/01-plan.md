# 🎯 Week 1 — Python for AI & LLM Fundamentals

> **Status:** ✅ COMPLETE (Batch 0, main-built template week)
> **Weekly Build:** **The AI Gateway** — a streaming, non-blocking chat backend with real-time token usage and cost tracking.

## Objective
Build your first LLM-powered app: a **streaming, non-blocking chat backend** that tracks
token usage and cost in real time. Along the way you internalize the Python and LLM
internals that every later week depends on.

## What You'll Learn
1. **Python fundamentals for AI** — functions, classes, objects, and the idioms that show up
   constantly in ML/AI code (dataclasses, context managers, type hints).
2. **Async Python, APIs & Streaming Responses** — `async/await`, generators, Server-Sent Events
   (SSE), and why LLM backends must be non-blocking.
3. **Transformer internals** — tokenization, vectorization (embeddings), and the attention
   mechanism (Q/K/V), explained with working from-scratch code.
4. **End-to-end LLM lifecycle & Model Tiering** — how a prompt becomes tokens becomes logits
   becomes text; and how to pick the right model tier for cost/latency/quality.

## Tools / Stack
- **Python 3.10–3.12** (async, dataclasses, type hints)
- **FastAPI** + **uvicorn** (non-blocking web framework)
- **httpx** (async HTTP client for OpenAI-compatible streaming)
- **Transformers concepts** (tokenization, attention) — implemented from scratch in numpy
- **Yolo-Auto** cloud LLM (OpenAI-compatible) via `shared/config.py`

## Weekly Build — The AI Gateway
A streaming, non-blocking chat backend with real-time token usage and cost tracking.

**Outcome:** Build your first LLM-powered app with a streaming backend and cost tracking.

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (6)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` — **The AI Gateway**: FastAPI + SSE streaming + token/cost tracking (framework-based)
- [x] `code/approach_2_raw_streaming_client.py` — raw httpx async streaming, manual SSE parse (no LangChain/FastAPI)
- [x] `code/approach_3_tokenizer_attention_from_scratch.py` — BPE-style tokenization + scaled dot-product attention in numpy (no API)
- [x] `code/model_tier_benchmark.py` — head-to-head latency/cost across model tiers
- [x] All scripts: Windows UTF-8 guard + graceful degradation + `py_compile` clean

## Prerequisites
- Any coding language (course assumes you can code; Python refresher is included here).
- `uv sync` done in `ai-engineering/`; `YOLO_AUTO_API_KEY` set (cloud LLM). **No Ollama** (8 GB laptop).

## How This Connects Forward
- **Async + streaming** → reused in Weeks 4 (agents), 8 (supervisor), 9 (multimodal).
- **Token/cost tracking** → becomes **unit economics** in Week 5 (evals & observability).
- **Model tiering** → drives **fine-tuning vs prompting vs RAG** decisions in Week 7.
- **Attention/embeddings** → foundation for **RAG** in Weeks 2–3.

## Time Estimate
~5 hours hands-on (learning 2h + build 2h + experiments 1h).
