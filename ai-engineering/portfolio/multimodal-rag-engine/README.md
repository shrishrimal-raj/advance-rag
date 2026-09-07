# The Multimodal RAG Engine

A **multimodal retrieval engine** that routes **text / image / audio** inputs through
modality processors, **fuses** their evidence into a single retrieval context, and tracks
**live dashboard metrics**. Week 9's weekly build from the SDE → AI Engineer course,
packaged as a portfolio service.

## What it does
- **Modality routing**: each input is classified (`text` / `image` / `audio`) and sent to
  the right processor; unknown types fall back to text.
- **Evidence fusion**: all modalities' extracted content is combined into one ranked
  retrieval context for grounding an answer.
- **Live dashboards**: per-modality counts, error count, and success rate — a snapshot any
  dashboard can poll.
- **Resilience**: a bad media file (decode error) is counted as an error and skipped — it
  never kills the whole query.

## Quick start
```bash
cd ai-engineering/portfolio/multimodal-rag-engine
pip install -e . pytest
pytest -q                 # offline, no CLIP/Whisper needed
uvicorn multimodal_rag_engine.app:app --port 8000
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"items":[{"type":"text","text":"what is this?"},{"type":"image","src":"chart.png"}]}'
curl localhost:8000/dashboard
```

## Layout
```
multimodal_rag_engine/
  core.py   # Evidence, DashboardMetrics, detect_modality, MultimodalRAGEngine
  app.py    # FastAPI wrapper (/health /ready /query /dashboard)
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Production notes
Back the processors with real models — CLIP for images, Whisper for audio, a text embedder
for text — then feed the fused context to your LLM. The routing + fusion + metrics contract
is unchanged by that swap. Point a Streamlit/Grafana dashboard at `/dashboard` for live ops.
