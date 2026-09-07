# PLANNING — The Regression Gate

## Problem
LLM/RAG changes silently degrade quality. Without an automated gate, regressions ship to
production. We need a CI check that scores a golden set and blocks merges when quality drops.

## Goals
1. Deterministic metrics over a golden dataset (offline, reproducible).
2. Baseline comparison with a configurable regression threshold.
3. Clear PASS/FAIL verdict + per-metric regression detail.
4. Wire into CI to block merges on FAIL.
5. Fully offline-testable (no LLM, no LangSmith).

## Non-goals
- Real RAGAS/LangSmith calls (heuristic stand-ins; swap-in point documented).
- Baseline management UI (baseline is a stored artifact).
- Statistical significance testing (threshold-based for v1).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | Tokenizer + 3 heuristic metrics | ✅ |
| M2 | score_dataset (mean per metric) | ✅ |
| M3 | run_gate baseline compare + threshold | ✅ |
| M4 | GateResult serialization | ✅ |
| M5 | FastAPI + Docker + CI | ✅ |

## Risks & mitigations
- Heuristic ≠ true quality → documented; swap in RAGAS for production fidelity.
- Threshold too loose/tight → configurable per gate call.
