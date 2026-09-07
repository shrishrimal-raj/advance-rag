# PLANNING — AI Gateway Service

## Problem
Applications call LLM providers that are flaky, slow, or rate-limited. A single
provider dependency is a reliability risk. We need a gateway that hides provider
instability behind one stable interface.

## Goals
1. Route a prompt across ordered providers with retry + fallback.
2. Protect upstream with a token-bucket rate limit.
3. Emit structured logs for every routing decision.
4. Degrade gracefully (503) when all providers are down.
5. Be fully testable offline (fake providers).

## Non-goals
- Prompt caching, semantic routing, or cost accounting (v2).
- Multi-tenant auth (v2).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | Core gateway (retry/fallback/rate-limit/log) | ✅ |
| M2 | Offline unit tests | ✅ |
| M3 | FastAPI wrapper + health/ready | ✅ |
| M4 | Docker + compose + CI | ✅ |
| M5 | Deployment runbook | ✅ |

## Risks & mitigations
- Provider SDK drift → providers are plain callables, so swapping is trivial.
- Rate-limit too aggressive → configurable `rate_limit_per_min`.
