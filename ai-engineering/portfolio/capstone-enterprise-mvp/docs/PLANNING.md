# PLANNING — The Capstone

## Problem
Individual skills (agents, RAG, streaming, CI) are only proven when they work together in
one deployable system. The capstone assembles them into a full-stack AI application.

## Goals
1. Multi-agent backend: ordered graph of planner → researcher → writer → critic.
2. RAG pipeline: researcher node pulls context from a pluggable retriever.
3. Streaming UI: emit per-node events over Server-Sent Events.
4. CI/CD: offline test suite gated in GitHub Actions.
5. Fully offline-testable core (pluggable callables; no providers required).

## Non-goals
- Real LLM/vector clients (pluggable stand-ins; swap-in points documented).
- Front-end framework (SSE endpoint is the contract; any client can consume it).
- LangSmith wiring (env + note; enabled in prod, not exercised offline).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | State + AgentGraph (run + stream) | ✅ |
| M2 | build_capstone_graph (4 nodes) | ✅ |
| M3 | FastAPI /ask + /ask/stream (SSE) | ✅ |
| M4 | Docker + Compose + CI | ✅ |
| M5 | Offline test suite | ✅ |

## Risks & mitigations
- Node ordering bugs → tests assert exact event sequence across all 4 nodes.
- Streaming backpressure → per-node yield keeps memory flat; no buffering of full output.
- Provider coupling → every external dependency is a pluggable callable.
