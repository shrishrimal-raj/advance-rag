# The Knowledge Engine

A **production RAG pipeline** with **hybrid retrieval** (BM25 sparse + dense cosine,
fused by Reciprocal Rank Fusion), **metadata filtering**, and **citation tracking**.
Week 2's weekly build from the SDE → AI Engineer course, packaged as a portfolio service.

## What it does
- Indexes document chunks (text + metadata + vector).
- Retrieves via hybrid BM25 + dense cosine, fused with RRF.
- Filters candidates by metadata (e.g. `topic=db`) before ranking.
- Returns ordered results with per-hit citations (`id`, `source`, `position`).
- Builds an LLM-ready prompt context + citation list.

## Quick start
```bash
cd ai-engineering/portfolio/knowledge-engine
pip install -e . pytest
pytest -q                 # offline, no network / no pgvector needed
uvicorn knowledge_engine.app:app --port 8000
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"database caching redis","top_k":3}'
```

## Layout
```
knowledge_engine/
  core.py   # Chunk, KnowledgeEngine (hybrid retrieve + filter + citations), cosine, tokenize
  app.py    # FastAPI wrapper (/health /ready /search /answer)
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml (with pgvector), .github/workflows/ci.yml
```

## Production notes
The core is dependency-light (pure-python + optional `rank-bm25`) so CI is green with no
secrets. In production, swap the seed corpus for pgvector storage and the stand-in vectors
for real embedding-model output — the `KnowledgeEngine` API is unchanged.
