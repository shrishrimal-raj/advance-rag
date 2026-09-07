# AI Gateway Service

A production-grade AI gateway that routes completion requests across multiple LLM
providers with **retry**, **fallback**, **token-bucket rate limiting**, and
**structured logging**. If every provider fails it degrades gracefully (503) instead
of crashing. Built as the flagship portfolio project for the SDE AI Engineer course.

## What it does
- Tries providers in priority order, retrying each up to `max_retries`.
- Falls back to the next provider on any failure.
- Enforces a per-minute token-bucket rate limit.
- Emits structured JSON log events for every decision.
- Exposes `/health`, `/ready`, and `/complete` over FastAPI.

## Quick start
```bash
cd ai-engineering/portfolio/ai-gateway-service
pip install -e . pytest
pytest -q                 # offline tests (fake providers, no network)
# run the API:
export LLM_API_KEY=sk-...
uvicorn ai_gateway.app:app --port 8000
curl -s localhost:8000/health
curl -s -X POST localhost:8000/complete -H 'content-type: application/json' -d '{"prompt":"hi"}'
```

## Layout
```
ai_gateway/
  gateway.py   # core: AIGateway, TokenBucket, GatewayConfig, GatewayResult
  app.py       # FastAPI wrapper (/health /ready /complete)
tests/
  test_gateway.py  # offline unit tests (fake providers)
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Design
See `docs/DESIGN.md` for architecture diagrams. See `docs/DEPLOYMENT.md` for rollout.
