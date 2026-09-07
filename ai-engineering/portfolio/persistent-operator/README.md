# The Persistent Operator

A **stateful tool-calling agent** that persists its conversation and tool-call history
to an external store (Redis in production) so it **survives process restarts**.
Week 4's weekly build from the SDE → AI Engineer course, packaged as a portfolio service.

## What it does
- Keeps per-session state (messages + tool calls) **outside the process**, keyed by
  session id in a store (Redis in prod, in-memory for tests).
- Registers and invokes tools; records every call + result.
- A new instance with the same store + session id resumes exactly where the old one left off.
- Sessions are isolated from each other.

## Quick start
```bash
cd ai-engineering/portfolio/persistent-operator
pip install -e . pytest
pytest -q                 # offline, no Redis needed
uvicorn persistent_operator.app:app --port 8000
curl -X POST localhost:8000/sessions/demo/message -H 'content-type: application/json' -d '{"message":"hi"}'
curl localhost:8000/sessions/demo/history
```

## Layout
```
persistent_operator/
  core.py   # MemoryStore, PersistentOperator (externalized state + tool calling)
  app.py    # FastAPI wrapper (/health /ready /sessions/{id}/message /sessions/{id}/history)
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml (with Redis), .github/workflows/ci.yml
```

## Production notes
The core uses `MemoryStore` so CI runs offline. In production, back it with Redis
(`REDIS_URL`) — the `PersistentOperator` API is unchanged; only the store implementation swaps.
