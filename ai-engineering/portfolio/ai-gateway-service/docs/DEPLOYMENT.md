# DEPLOYMENT — AI Gateway Service

## Local
```bash
pip install -e .
export LLM_API_KEY=sk-...
uvicorn ai_gateway.app:app --port 8000
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
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q` on push/PR.
Tests are offline (fake providers), so CI needs no secrets or network.

## Rollout checklist
- [ ] Set `LLM_API_KEY` in the target environment.
- [ ] Verify `/ready` returns `{"ready": true}`.
- [ ] Confirm `/health` 200 and `/complete` 200 on a smoke prompt.
- [ ] Watch structured logs for `provider_error` / `all_providers_failed` spikes.
- [ ] Tune `GATEWAY_RATE_LIMIT_PER_MIN` to your provider quota.
