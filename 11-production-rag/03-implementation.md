# 🔨 Module 11 — Implementation: Building the Production Tools

All commands run from the **project root** (`D:\Raj\Projects\advance-rag`) with `uv`.
Each code file is **standalone and runnable** — no LLM required (they use local embeddings only).

> **Prereq:** `uv sync` done. Local embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
> auto-download on first run (~90 MB). No API key or Ollama needed for these three tools.

---

## Tool 1 — Semantic Cache (`code/semantic_cache.py`)

**What it does:** a `SemanticCache` class backed by a Chroma "cache" collection. For a query it
embeds the text, searches for the nearest stored entry, and if cosine similarity > threshold
(0.95) returns the cached answer (**HIT**); otherwise it runs the wrapped generator function,
stores the result, and returns it (**MISS**).

**Key ideas to notice:**
- Chroma cosine *distance* = `1 - similarity`, so a hit means `distance < 1 - threshold`.
- The cache *wraps* a generator function — you never change how answers are produced, only add
  a fast path in front of it.
- Invalidation: `clear()` / versioned collections (see learning doc §2).

**Run it:**
```bash
uv run python 11-production-rag/code/semantic_cache.py
```
**Expected:** 3 queries; #3 is a paraphrase of #1 → prints **HIT** with the similarity score.

---

## Tool 2 — Guardrails (`code/guardrails.py`)

**What it does:** a `GuardrailPipeline` running input checks then output checks.
- **Input validators:** blocked-topics regex list, prompt-injection pattern detection
  ("ignore previous instructions", role-hijack phrases), length limits.
- **Output validators:** refusal detection, PII regex redaction (emails + phone numbers),
  toxicity keyword check, citation-presence check.

**Key ideas:**
- Input failures **block** the request (return a reason, don't call the LLM).
- Output failures **sanitize** (redact PII) or flag (toxicity / missing citations).
- No LLM involved — pure text rules, so it's fast and deterministic.

**Run it:**
```bash
uv run python 11-production-rag/code/guardrails.py
```
**Expected:** a malicious "ignore previous instructions…" query is **blocked**; a normal answer
containing an email + phone gets **PII redacted**.

---

## Tool 3 — Monitoring (`code/monitoring.py`)

**What it does:** a `PipelineMetrics` dataclass plus a context-manager/decorator that records
per-request per-stage latency, token counts, cache hits, and errors. It aggregates into a stats
report (mean/p95 latency, hit rate, error rate) printed as a rich table.

**Key ideas:**
- A `@timed("stage")` decorator and a `with metrics.stage("retrieve"):` context manager both feed
  the same collector — pick whichever fits your code.
- p95 is computed from the recorded latencies; hit-rate = cache_hits / total; error-rate = errors / total.
- The demo simulates 20 requests with random latencies so you see a real report without traffic.

**Run it:**
```bash
uv run python 11-production-rag/code/monitoring.py
```
**Expected:** a rich table with mean/p95 latency per stage, overall hit rate, and error rate over
20 simulated requests.

---

## Build order & verification

| Step | Command | Verify |
|------|---------|--------|
| 1 | `.../code/semantic_cache.py` | #3 query shows HIT + similarity |
| 2 | `.../code/guardrails.py` | injection blocked; PII redacted |
| 3 | `.../code/monitoring.py` | mean/p95/hit-rate/error-rate table |

## How these plug into Papeer (Module 10)
- Wrap Papeer's `retrieve()`/`run_agent()` with the **semantic cache** in front of generation.
- Put the **guardrail pipeline** at the FastAPI boundary (input before, output after).
- Instrument every stage with the **monitoring** collector; expose `/metrics` on the API.

## Troubleshooting
- **Embedding download slow:** first run fetches MiniLM; cached afterwards.
- **Semantic cache MISS when you expected HIT:** lower the threshold slightly or use a closer
  paraphrase; the tool prints the actual similarity so you can tune.
- **Chroma lock error:** close other processes using the same persist dir, or use a distinct
  collection name.
