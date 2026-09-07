# The Dual-Agent Supervisor

A **secure maker-checker multi-agent system** with **per-agent provider failover** that
survives a live provider-outage recovery drill. Week 8's weekly build from the SDE → AI
Engineer course, packaged as a portfolio service.

## What it does
- **Maker-checker**: the maker agent produces an output; the checker agent validates it
  against a policy. On rejection, the maker retries (bounded).
- **Failover**: each agent is wrapped with a primary + fallback provider. If the primary
  raises (outage), the call transparently fails over to the fallback — the system keeps
  working through a provider outage.
- **Tracing**: every attempt is recorded (maker value, provider used, verdict) for audit.

## Quick start
```bash
cd ai-engineering/portfolio/dual-agent-supervisor
pip install -e . pytest
pytest -q                 # offline, no providers needed
uvicorn dual_agent_supervisor.app:app --port 8000
curl -X POST localhost:8000/run -H 'content-type: application/json' -d '{"task":"draft a refund policy"}'
```

## Layout
```
dual_agent_supervisor/
  core.py   # ProviderOutage, CheckerVerdict, make_failover, DualAgentSupervisor
  app.py    # FastAPI wrapper (/health /ready /run)
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Production notes
Wire `make_failover` around real provider clients (OpenAI primary → Anthropic fallback,
etc.). The checker's policy is where you encode security/compliance rules (PII checks,
guardrails, schema validation). The outage-recovery drill is exactly `test_failover_on_primary_outage`:
primary raises, fallback serves, system stays up.
