# The Capstone

A **full-stack AI application** that ties the whole course together: a **multi-agent
backend** (LangGraph-style planner → RAG researcher → writer → critic), a **RAG pipeline**,
a **streaming UI** (Server-Sent Events), and a **CI/CD pipeline**. Week 9's capstone from
the SDE → AI Engineer course, packaged as a portfolio service.

## What it does
- **Multi-agent graph**: an ordered `AgentGraph` runs named nodes over a shared `State`.
- **RAG node**: the researcher pulls context via a pluggable retriever.
- **Streaming**: `/ask/stream` emits Server-Sent Events as each node completes — a live UI
  can render progress token-by-token / step-by-step.
- **CI/CD**: GitHub Actions runs the offline test suite on every push/PR.

## Quick start
```bash
cd ai-engineering/portfolio/capstone
pip install -e . pytest
pytest -q                 # offline, no providers needed
uvicorn capstone.app:app --port 8000
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"query":"explain RAG"}'
curl -N -X POST localhost:8000/ask/stream -H 'content-type: application/json' -d '{"query":"explain RAG"}'
```

## Layout
```
capstone/
  core.py   # State, AgentGraph, build_capstone_graph (planner/researcher/writer/critic)
  app.py    # FastAPI wrapper (/health /ready /ask /ask/stream SSE)
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Production notes
Swap the pluggable nodes for real LLM clients and a real vector-store retriever; enable
LangSmith tracing (`LANGSMITH_TRACING=true`) for full observability of every agent step;
serve the SSE stream to a React/Vite front end. The graph contract (`State` in → `State`
out, events streamed) is unchanged by any of those swaps.
