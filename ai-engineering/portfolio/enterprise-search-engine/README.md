# The Enterprise Search Engine

A **multi-tenant search API** with **hybrid retrieval** (BM25 sparse + dense cosine,
fused by RRF), **metadata filtering**, and **citation tracking**. Week 3's weekly build
from the SDE → AI Engineer course, packaged as a portfolio service.

## What it does
- **Multi-tenant isolation**: each tenant is an isolated namespace (Pinecone namespace in
  production). A query from tenant A can never return tenant B's documents.
- Hybrid BM25 + dense cosine retrieval, fused by Reciprocal Rank Fusion.
- Metadata filtering scoped within the tenant.
- Ordered results with per-hit citations.

## Quick start
```bash
cd ai-engineering/portfolio/enterprise-search-engine
pip install -e . pytest
pytest -q                 # offline, no Pinecone needed
uvicorn enterprise_search.app:app --port 8000
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -H 'X-Tenant-ID: demo' -d '{"query":"rate limits","top_k":3}'
```

## Layout
```
Enterprise_search/
  core.py   # Doc, EnterpriseSearchEngine (multi-tenant hybrid), cosine, tokenize
  app.py    # FastAPI wrapper (/health /ready /search) — tenant via X-Tenant-ID header
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Production notes
The core emulates Pinecone namespaces with an in-memory dict so CI runs offline with no
secrets. In production, map each `tenant_id` to a Pinecone namespace and swap stand-in
vectors for real embeddings — the `EnterpriseSearchEngine` API is unchanged.
