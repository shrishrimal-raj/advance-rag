# PLANNING — Regression Telemetry Gate

## Problem
AI systems silently regress: a prompt tweak or model swap can drop answer quality
without any test failing. We need an automated gate that measures quality, compares
it to a baseline, and blocks regressions.

## Goals
1. Run evals over a dataset with a pluggable scorer.
2. Aggregate mean score and pass rate.
3. Detect regression against a stored baseline.
4. Enforce a minimum pass rate.
5. Emit structured telemetry for observability.
6. Fully testable offline.

## Non-goals
- Storing/managing baselines across runs (v2 — add a baseline store).
- Statistical significance testing (v2).
- Multi-metric dashboards (v2).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | Pluggable scorer + run_evals | ✅ |
| M2 | Metrics aggregation | ✅ |
| M3 | Baseline compare + regression detect | ✅ |
| M4 | Telemetry emit | ✅ |
| M5 | Offline tests + API + Docker + CI | ✅ |

## Risks & mitigations
- Scorer noise → deterministic default; swap in a robust judge for production.
- Threshold too tight → configurable `regression_threshold` / `min_pass_rate`.
