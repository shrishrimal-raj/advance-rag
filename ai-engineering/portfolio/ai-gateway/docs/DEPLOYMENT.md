# DEPLOYMENT — The AI Gateway

## Local
```bash
pip install -e .
export LLM_API_KEY=sk-...
uvicorn ai_gateway.app:app --port 8000
curl -N -X POST localhost:8000/chat/stream -H 'content-type: application/json' -d '{"prompt":"hi"}'
```

## Docker
```bash
docker build -t ai-gateway .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... ai-gateway
```

## Compose
```bash
cp .env.example .env   # fill LLM_API_KEY
docker compose up -d
curl localhost:8000/health
curl localhost:8000/usage
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are
offline (fake async provider) — no secrets or network required.

## Rollout checklist
- [ ] Set `LLM_API_KEY` (+ `LLM_BASE_URL`) in the target environment.
- [ ] Verify `/ready` reports `{"ready": true}` and the model catalog.
- [ ] Smoke `/chat/stream`; confirm SSE `token` events then a final `usage` event.
- [ ] Watch `/usage` for live cost accumulation.
