# PLANNING — The Dual-Agent Supervisor

## Problem
A single LLM agent is both a reliability and a safety risk: one provider outage takes it
down, and unreviewed outputs can be wrong or unsafe. A secure system needs a second
(checker) agent and redundancy across providers.

## Goals
1. Maker-checker loop: maker produces, checker validates against a policy.
2. Bounded retries on rejection (no infinite loops).
3. Per-agent failover: primary provider outage → fallback serves.
4. Full trace of every attempt for audit.
5. Fully offline-testable (pluggable callables; no real providers).

## Non-goals
- Real LLM clients (pluggable callables stand in; swap-in point documented).
- Distributed state / persistence (in-memory trace).
- Complex orchestration graphs (two agents, one loop — deliberately minimal).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | make_failover (primary→fallback) | ✅ |
| M2 | DualAgentSupervisor maker-checker loop | ✅ |
| M3 | Bounded retries + exhaustion | ✅ |
| M4 | Outage failover drill (test) | ✅ |
| M5 | FastAPI + Docker + CI | ✅ |

## Risks & mitigations
- Infinite retry loop → hard `max_retries` bound; test asserts exhaustion.
- Silent failover masking outages → trace records which provider served each attempt.
- Unsafe output shipped → checker policy gates every output before approval.
