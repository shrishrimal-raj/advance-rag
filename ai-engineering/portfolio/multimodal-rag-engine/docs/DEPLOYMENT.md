# DEPLOYMENT — The Multimodal RAG Engine

## Local (offline)
```bash
pip install -e .
uvicorn multimodal_rag_engine.app:app --port 8000
curl -X POST localhost:8000/query -H 'content-type: application/json' -d '{"items":[{"type":"text","text":"what is this?"}]}'
curl localhost:8000/dashboard
```

## Docker
```bash
docker build -t multimodal-rag-engine .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... -e WHISPER_MODEL=tiny multimodal-rag-engine
```

## Compose
```bash
cp .env.example .env   # fill keys + model names
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "modalities": ["text", "image", "audio"]}
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are fully
offline (pluggable processors) — no API keys, no CLIP/Whisper downloads required.

## Production rollout
- [ ] Back image processor with CLIP (`CLIP_MODEL`), audio with Whisper (`WHISPER_MODEL`).
- [ ] Feed fused context to the LLM for grounded answers.
- [ ] Point a Streamlit/Grafana dashboard at `/dashboard` (poll every `DASHBOARD_REFRESH_MS`).
- [ ] Persist processed items to the vector store for cross-query recall.
