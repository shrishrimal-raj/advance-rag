# 🔨 Week 1 Implementation — Build The AI Gateway

> Step-by-step build. Follow in order; each step ends with a **verify** line.
> All commands run from `ai-engineering/`.

## 0. Setup
```bash
cd ai-engineering
uv sync                 # install deps (fastapi, uvicorn, httpx, ...)
cp ../.env ./.env       # reuse repo-root .env (has YOLO_AUTO_API_KEY)
```
**Verify:** `uv run python -c "import fastapi, httpx; print('ok')"` → `ok`

## 1. The AI Gateway (`code/main.py`) — framework-based
A FastAPI app that streams chat tokens over SSE and tracks usage + cost.

Key design points:
- `create_app()` factory imports FastAPI lazily → module stays importable without a server.
- `POST /chat` returns an SSE stream: one event per token, final event carries `{usage, cost}`.
- `GET /health` for probes.
- **Cost tracking:** a small `CostTracker` dataclass accumulates prompt/completion tokens and
  computes cost from a per-tier price table.
- **Graceful degradation:** no API key → the endpoint returns a clear JSON hint (HTTP 200 with
  `error` field) instead of crashing; CLI `--selftest` prints the same hint.

**Verify (no server needed):**
```bash
uv run python 01-python-llm-fundamentals/code/main.py --selftest
# -> makes ONE streaming call via httpx, prints tokens + cost (or a graceful "no key" hint)
```

**Verify (real server):**
```bash
uv run uvicorn main:create_app --factory --app-dir 01-python-llm-fundamentals/code --port 8000
# in another shell:
curl -N -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"message":"Say hi in 3 words"}'
```

## 2. Approach 2 — raw streaming client (`code/approach_2_raw_streaming_client.py`)
No LangChain, no FastAPI. Just `httpx.AsyncClient` hitting the OpenAI-compatible
`/chat/completions` endpoint with `stream=true`, manually parsing the SSE lines
(`data: {...}`, terminal `data: [DONE]`). Prints each token as it arrives + final usage/cost.

**Why this approach matters:** it proves you understand the wire protocol, not just the wrapper.

**Verify:**
```bash
uv run python 01-python-llm-fundamentals/code/approach_2_raw_streaming_client.py
```

## 3. Approach 3 — tokenizer + attention from scratch (`code/approach_3_tokenizer_attention_from_scratch.py`)
Pure numpy/python, **no API calls** (fully local, light on 8 GB):
- A tiny byte-level BPE-style tokenizer: text → token ids → back (round-trip check).
- Scaled dot-product attention implemented by hand: build Q/K/V, compute
  `softmax(QKᵀ/√d)·V`, and show how attention weights shift when a "relevant" token is present.

**Why this approach matters:** it demystifies what the provider does internally — the core of the course's "understand, don't just call" ethos.

**Verify (no network):**
```bash
uv run python 01-python-llm-fundamentals/code/approach_3_tokenizer_attention_from_scratch.py
```

## 4. Model tier benchmark (`code/model_tier_benchmark.py`)
Makes a few identical requests across 2–3 model tiers (small vs large) and prints a table of
TTFT, total latency, estimated tokens, and estimated cost. Demonstrates the tiering trade-off
from the learning doc.

**8 GB note:** this makes a handful of cloud calls. Run with `--quick` (fewer iterations) to stay light.

**Verify:**
```bash
uv run python 01-python-llm-fundamentals/code/model_tier_benchmark.py --quick
```

## 5. Full verification sweep
```bash
cd ai-engineering
for f in 01-python-llm-fundamentals/code/*.py; do uv run python -m py_compile "$f" && echo "OK $f"; done
```
All four must print OK. Then run the two no-network scripts fully (approach_3 always; approach_2/benchmark if you want live LLM proof).

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `no YOLO_AUTO_API_KEY` hint | Copy `../.env` to `.env` or export the key |
| Slow first run | MiniLM/embeddings cache on first load; subsequent runs are fast |
| SSE looks like one blob in browser | Use `curl -N` (no buffering) or an EventSource client |
| Port in use | Change `--port` |
