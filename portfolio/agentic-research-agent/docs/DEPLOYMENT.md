# Deployment — Agentic Research Agent

## Docker

```bash
docker build -t ara .
docker run -p 8000:8000 -e YOLO_AUTO_API_KEY=... -v $(pwd)/chroma_db:/srv/chroma_db ara
```

## VPS (systemd)
Same pattern as the other projects: `uv sync --no-dev`, systemd unit running `uvicorn main:app --host 127.0.0.1 --port 8000` with `EnvironmentFile=.env`, TLS via reverse proxy.

## Notes
- Ingest the corpus before first serve (`research_docs` collection must exist).
- Bound cost: the reflection loop multiplies LLM calls per question — set a per-IP rate limit at the proxy.