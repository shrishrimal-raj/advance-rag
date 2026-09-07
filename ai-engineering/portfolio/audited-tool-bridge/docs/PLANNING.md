# PLANNING — The Audited Tool Bridge

## Problem
Giving AI agents direct access to internal tools is dangerous: no access control, no
throttling, no record of what was called. A secure bridge must authenticate, rate-limit,
and audit every call.

## Goals
1. Authentication: reject calls without a valid API key.
2. Per-client rate limiting (token bucket).
3. Full audit log of every call (allowed or denied) with reason.
4. Tool registration + dispatch with error capture.
5. Fully offline-testable (in-memory auth + audit).

## Non-goals
- Real identity provider (API-key map stand-in; swap-in point documented).
- Durable audit sink (in-memory list; swap to Kafka/S3 in prod).
- MCP wire protocol framing (HTTP endpoints stand in for MCP tool calls).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | TokenBucket rate limiter | ✅ |
| M2 | ToolBridge auth + dispatch | ✅ |
| M3 | Rate limiting per client | ✅ |
| M4 | Audit logging + per-client filter | ✅ |
| M5 | FastAPI + Docker + CI | ✅ |

## Risks & mitigations
- Auth bypass → check key before anything else; test asserts rejection.
- Audit loss → append on every path (allow/deny/error); test covers each.
