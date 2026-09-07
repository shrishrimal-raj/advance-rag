# DEPLOYMENT — The Dual-Agent Supervisor

## Local (offline)
```bash
pip install -e .
uvicorn dual_agent_supervisor.app:app --port 8000
curl -X POST localhost:8000/run -H 'content-type: application/json' -d '{"task":"draft a refund policy"}'
```

## Docker
```bash
docker build -t dual-agent-supervisor .
docker run -p 8000:8000 -e LLM_API_KEY=sk-... dual-agent-supervisor
```

## Compose
```bash
cp .env.example .env   # fill keys + model tiers
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "pattern": "maker-checker + failover"}
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are fully
offline (pluggable callables) — no API keys or external services required.

## Outage-recovery drill
The drill is codified in `tests/test_core.py::test_failover_on_primary_outage`: the primary
raises `ProviderOutage`, the fallback serves, and the run still succeeds with
`maker_provider == "fallback"`. In production, point the primary/fallback at real provider
clients and re-run this same path during a maintenance window.
