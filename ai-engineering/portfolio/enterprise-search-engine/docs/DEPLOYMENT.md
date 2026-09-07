# DEPLOYMENT — The Enterprise Search Engine

## Local (offline)
```bash
pip install -e .
uvicorn enterprise_search.app:app --port 8000
curl -X POST localhost:8000/search -H 'content-type: application/json' -H 'X-Tenant-ID: demo' -d '{"query":"rate limits","top_k":3}'
```

## Docker
```bash
docker build -t enterprise-search .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... -e PINECONE_API_KEY=pc-... enterprise-search
```

## Compose
```bash
cp .env.example .env   # fill keys
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "tenants": ["demo"]}
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are
offline (namespace emulation, optional rank_bm25) — no secrets or Pinecone required.

## Rollout checklist
- [ ] Put an auth gateway in front that sets/validates `X-Tenant-ID`.
- [ ] Map each tenant to a real Pinecone namespace.
- [ ] Verify cross-tenant isolation under load (test asserts no leak).
- [ ] Confirm hybrid ranking + citations in `/search` responses.
