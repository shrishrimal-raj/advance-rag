# DEPLOYMENT — The Audited Tool Bridge

## Local (offline)
```bash
pip install -e .
uvicorn audited_tool_bridge.app:app --port 8000
curl -X POST localhost:8000/tools/sum -H 'content-type: application/json' -H 'X-API-Key: demo-key' -d '{"args":{"a":2,"b":3}}'
curl localhost:8000/audit
```

## Docker
```bash
docker build -t audited-tool-bridge .
docker run -p 8000:8000 -e BRIDGE_API_KEYS=demo-key=demo-client audited-tool-bridge
```

## Compose
```bash
cp .env.example .env   # fill keys
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "tools": ["echo", "sum"]}
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are
offline (in-memory auth + audit) — no secrets or external services required.

## Rollout checklist
- [ ] Back `BRIDGE_API_KEYS` with real issued keys (rotate regularly).
- [ ] Ship audit entries to a durable sink (`AUDIT_SINK`).
- [ ] Tune `RATE_CAPACITY` / `RATE_REFILL_PER_SEC` per client tier.
- [ ] Expose as an MCP server so CrewAI agents call tools over the standard protocol.
