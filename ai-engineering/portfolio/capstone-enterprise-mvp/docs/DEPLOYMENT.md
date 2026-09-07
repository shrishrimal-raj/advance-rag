# DEPLOYMENT — The Capstone

## Local (offline)
```bash
pip install -e .
uvicorn capstone.app:app --port 8000
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"query":"explain RAG"}'
curl -N -X POST localhost:8000/ask/stream -H 'content-type: application/json' -d '{"query":"explain RAG"}'
```

## Docker
```bash
docker build -t capstone .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... -e LANGSMITH_API_KEY=lsm-... capstone
```

## Compose
```bash
cp .env.example .env   # fill keys + models
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "stack": "multi-agent + RAG + streaming"}
```

## CI/CD
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q` on every push to
main/master/develop and on PRs. Tests are fully offline — no API keys or external services.
To extend to CD, add a build+push image step after the test job passes.

## Production rollout
- [ ] Swap pluggable nodes for real LLM clients (planner/writer/critic tiers).
- [ ] Point the researcher at a real vector store (Chroma/pgvector).
- [ ] Enable LangSmith tracing (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`).
- [ ] Serve `/ask/stream` to a React/Vite front end for the live UI.
