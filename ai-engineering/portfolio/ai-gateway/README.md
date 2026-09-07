# The AI Gateway

A **streaming, non-blocking chat backend** with **real-time token usage and cost
tracking**. This is Week 1's weekly build from the SDE → AI Engineer course, packaged
as a production-grade portfolio service.

## What it does
- Streams LLM responses as Server-Sent Events (SSE) — non-blocking, token by token.
- Tracks input/output tokens and computes USD cost per request using a model price table.
- Aggregates usage across all requests in a live `TokenCostTracker`.
- Exposes `/health`, `/ready`, `/usage`, and `POST /chat/stream` over FastAPI.

## Quick start
```bash
cd ai-engineering/portfolio/ai-gateway
pip install -e . pytest
pytest -q                 # offline tests (fake async provider, no network)
# run the API:
export LLM_API_KEY=sk-...
uvicorn ai_gateway.app:app --port 8000
curl -N -X POST localhost:8000/chat/stream -H 'content-type: application/json' \
  -d '{"prompt":"hello","model":"gpt-4o-mini"}'
curl -s localhost:8000/usage
```

## Layout
```
ai_gateway/
  core.py   # stream_chat, TokenCostTracker, Usage, compute_cost, estimate_tokens
  app.py    # FastAPI wrapper (/health /ready /usage /chat/stream)
tests/
  test_core.py   # offline unit tests
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Design
See `docs/DESIGN.md` for architecture diagrams. See `docs/DEPLOYMENT.md` for rollout.
