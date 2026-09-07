# DEPLOYMENT — The Knowledge Engine

## Local (offline)
```bash
pip install -e .
uvicorn knowledge_engine.app:app --port 8000
curl -X POST localhost:8000/search -H 'content-type: application/json' -d '{"query":"redis caching","top_k":3}'
```

## Docker (app only)
```bash
docker build -t knowledge-engine .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... knowledge-engine
```

## Compose (app + pgvector)
```bash
cp .env.example .env   # fill LLM_API_KEY, PGVECTOR_DSN
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "chunks": N}
```
`docker-compose.yml` starts `pgvector/pgvector:pg16` alongside the app; point
`PGVECTOR_DSN` at it once you wire real storage.

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are
offline (stand-in vectors, optional rank_bm25) — no secrets or database required.

## Rollout checklist
- [ ] Load real corpus into pgvector; replace seed corpus.
- [ ] Wire embedding model for query + doc vectors.
- [ ] Verify `/search` returns hybrid-ranked hits with citations.
- [ ] Confirm metadata filters scope results correctly.
