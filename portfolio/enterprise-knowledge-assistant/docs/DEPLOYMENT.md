# Deployment — Enterprise Knowledge Assistant

## Docker

```bash
docker build -t ekb-rag .
docker run -p 8000:8000 -e YOLO_AUTO_API_KEY=... -v $(pwd)/data:/srv/data ekb-rag
```

## VPS (systemd)

```bash
uv venv && uv sync --no-dev
# /etc/systemd/system/ekb-rag.service
[Unit]
Description=Enterprise Knowledge Assistant
After=network.target
[Service]
WorkingDirectory=/opt/ekb-rag
ExecStart=/opt/ekb-rag/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
EnvironmentFile=/opt/ekb-rag/.env
Restart=on-failure
[Install]
WantedBy=multi-user.target
```
Put nginx/caddy in front for TLS. Back up `chroma_db/` nightly (it is the source of truth).

## Rollback
Keep the previous image tag; `docker tag` swap + restart is a <1 min rollback. Knowledge-base changes are additive — re-ingestion is idempotent (stable chunk IDs).