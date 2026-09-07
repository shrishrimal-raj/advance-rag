# DEPLOYMENT — Enterprise Search Service

## Local
```bash
pip install -e .
uvicorn enterprise_search.app:app --port 8000
curl -s localhost:8000/health
```

## Docker
```bash
docker build -t enterprise-search .
docker run -p 8000:8000 enterprise-search
```

## Compose
```bash
cp .env.example .env
docker compose up -d
curl -s -X POST localhost:8000/search -H 'content-type: application/json' -d '{"query":"password reset"}'
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are
offline (in-memory corpus, deterministic scorers) — no secrets or network required.

## Rollout checklist
- [ ] Verify `/ready` reports the indexed doc count.
- [ ] Smoke `/search` with a known query; confirm expected doc ranks first.
- [ ] If using real embeddings, set `EMBEDDING_MODEL` and confirm the swap-in works.
- [ ] Tune `SEARCH_TOP_K` and RRF `k` against your relevance eval set.
