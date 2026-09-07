# The Audited Tool Bridge

A **secure MCP-style tool gateway** that exposes internal tools to AI agents with
**authentication**, per-client **rate limiting**, and **full audit logging**.
Week 6's weekly build from the SDE → AI Engineer course, packaged as a portfolio service.

## What it does
- Registers internal tools and exposes them to agents behind a gateway.
- **Auth**: every call requires a valid API key; invalid keys are rejected.
- **Rate limiting**: per-client token bucket; exhausted clients are throttled.
- **Audit logging**: every call (allowed or denied) is recorded with timestamp, client,
  tool, args, and outcome — queryable per client.

## Quick start
```bash
cd ai-engineering/portfolio/audited-tool-bridge
pip install -e . pytest
pytest -q                 # offline, no external services needed
uvicorn audited_tool_bridge.app:app --port 8000
curl -X POST localhost:8000/tools/sum -H 'content-type: application/json' \
  -H 'X-API-Key: demo-key' -d '{"args":{"a":2,"b":3}}'
curl localhost:8000/audit
```

## Layout
```
audited_tool_bridge/
  core.py   # TokenBucket, AuditEntry, ToolBridge (auth -> rate limit -> dispatch -> audit)
  app.py    # FastAPI wrapper (/health /ready /tools/{name} /audit)
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Production notes
The core uses in-memory auth + audit so CI runs offline. In production, back auth with your
identity provider and ship audit entries to a durable sink (Kafka/S3/LangSmith) — the
`ToolBridge.invoke` contract (auth → rate limit → dispatch → audit) is unchanged. This is the
MCP server half of an MCP/CrewAI integration; CrewAI agents call these tools over HTTP.
