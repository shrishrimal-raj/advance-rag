# PLANNING — The AI Gateway

## Problem
LLM calls are expensive and slow. A chat backend must stream tokens as they arrive
(non-blocking) and account for what each request costs in real time, or you cannot
manage spend or latency.

## Goals
1. Stream responses token-by-token over SSE without blocking the event loop.
2. Track input/output tokens per request.
3. Compute USD cost per request from a model price table.
4. Aggregate usage/cost across all requests (live accounting).
5. Be fully testable offline (fake async provider).

## Non-goals
- Prompt caching, semantic routing, multi-tenant auth (v2).
- Real tokenizer (we use a ~4 chars/token heuristic; swap in tiktoken for exact counts).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | Cost model + token estimation | ✅ |
| M2 | Streaming chat (async generator) | ✅ |
| M3 | Real-time usage tracker | ✅ |
| M4 | FastAPI SSE endpoint + tests | ✅ |
| M5 | Docker + compose + CI | ✅ |

## Risks & mitigations
- Token estimate imprecise → documented heuristic; swap-in point for tiktoken.
- Provider SDK drift → provider is an async callable, trivially swappable.
