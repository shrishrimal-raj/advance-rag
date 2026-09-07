# DEPLOYMENT — The Persistent Operator

## Local (offline)
```bash
pip install -e .
uvicorn persistent_operator.app:app --port 8000
curl -X POST localhost:8000/sessions/demo/message -H 'content-type: application/json' -d '{"message":"hi"}'
curl localhost:8000/sessions/demo/history
```

## Docker
```bash
docker build -t persistent-operator .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... -e REDIS_URL=redis://localhost:6379/0 persistent-operator
```

## Compose (app + Redis)
```bash
cp .env.example .env   # fill keys
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "tools": ["echo", "upper"]}
```
`docker-compose.yml` starts `redis:7-alpine` alongside the app; point `REDIS_URL` at it
and swap `MemoryStore` for a redis-backed store once you go live.

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests use the
in-memory store — no Redis or secrets required.

## Rollout checklist
- [ ] Back the store with Redis (`REDIS_URL`); add TTL via `SESSION_TTL_SECONDS`.
- [ ] Wire the real LLM/LangGraph `decide` loop.
- [ ] Verify restart continuity: kill the app, restart, confirm `/history` intact.
