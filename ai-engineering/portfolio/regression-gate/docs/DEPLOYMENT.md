# DEPLOYMENT — The Regression Gate

## Local (offline)
```bash
pip install -e .
uvicorn regression_gate.app:app --port 8000
curl -X POST localhost:8000/gate -H 'content-type: application/json' -d '{"cases":[{"question":"What is RAG?","answer":"RAG retrieves then generates.","context":"RAG retrieves documents then generates.","expected":"RAG retrieves then generates"}]}'
```

## Docker
```bash
docker build -t regression-gate .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... -e LANGSMITH_API_KEY=lsv-... regression-gate
```

## Compose
```bash
cp .env.example .env   # fill keys
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "metrics": [...]} 
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are
offline (heuristic metrics) — no secrets or LLM required. To block merges, add a step that
POSTs the golden set to `/gate` and exits non-zero when `passed == false`.

## Rollout checklist
- [ ] Store the golden baseline as a build artifact from the last green merge.
- [ ] Add a CI step that calls `/gate` and fails the job on regression.
- [ ] Swap heuristic metrics for RAGAS scorers; push traces to LangSmith.
