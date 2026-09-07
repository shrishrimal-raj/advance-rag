# PLANNING — The Persistent Operator

## Problem
Agents are stateful. If state lives only in process memory, a deploy or crash wipes the
conversation and any tool work already done. Production agents must persist state so they
resume cleanly.

## Goals
1. Externalize all agent state (messages + tool calls) to a store keyed by session id.
2. Tool calling with recorded call + result.
3. Survive restart: new instance + same store/session = full continuity.
4. Session isolation.
5. Fully offline-testable (in-memory store).

## Non-goals
- Real LLM decision loop (`decide` is injected; wire LangGraph/LLM in production).
- Real Redis client (MemoryStore stand-in; swap-in point documented).
- Multi-node coordination / locking.

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | Store abstraction (get/set) | ✅ |
| M2 | PersistentOperator load/save per op | ✅ |
| M3 | Tool calling + recording | ✅ |
| M4 | step() turn loop + session isolation | ✅ |
| M5 | FastAPI + Docker/compose(Redis) + CI | ✅ |

## Risks & mitigations
- State loss on crash → save after every mutation (write-through).
- Store swap → interface is 3 methods; Redis impl drops in.
