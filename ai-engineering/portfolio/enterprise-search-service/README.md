# Enterprise Search Service

A hybrid search engine that combines **BM25** (sparse/lexical) and **dense vector**
cosine retrieval, fuses them with **Reciprocal Rank Fusion (RRF)**, and optionally
**reranks** with a cross-encoder-style joint score. Exposed as a FastAPI service.
Built as a portfolio project for the SDE AI Engineer course.

## Why hybrid?
- BM25 nails exact terms, acronyms, rare identifiers.
- Dense vectors catch paraphrase and semantic similarity.
- RRF merges both rank lists robustly (no score-scale alignment needed).
- Reranking re-scores the small candidate set for final precision.

## Quick start
```bash
cd ai-engineering/portfolio/enterprise-search-service
pip install -e . pytest
pytest -q                 # offline tests, no network/model needed
# run the API:
uvicorn enterprise_search.app:app --port 8000
curl -s -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"connection pool exhausted","top_k":3}'
```

## Layout
```
Enterprise_search/
  search.py   # Index (bm25/dense/rrf/rerank), SearchService, tokenize
  app.py      # FastAPI wrapper (/health /ready /search)
tests/
  test_search.py   # offline unit tests
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Note on the dense path
The dense scorer uses a deterministic bag-of-words L2-normalized vector so it runs
anywhere with no model download. To use real embeddings, replace `Index._vec` with a
call to your embedding model (`EMBEDDING_MODEL` in `.env.example`). The rest of the
pipeline (RRF + rerank) is unchanged.
