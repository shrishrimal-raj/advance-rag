# Regression Telemetry Gate

A CI quality gate for AI systems. It runs evals over a dataset, aggregates metrics,
compares them against a stored **baseline**, detects **regression**, and emits
structured **telemetry**. Wire it into CI to fail the build when quality drops.
Built as a portfolio project for the SDE AI Engineer course.

## What it does
- Runs evals with a pluggable scorer (`scorer(answer, expected) -> float`).
- Aggregates `mean_score` and `pass_rate`.
- Fails if `pass_rate < min_pass_rate`.
- Fails if `mean_score` drops more than `regression_threshold` below baseline.
- Emits a JSON telemetry record (and can write it to a file).

## Quick start
```bash
cd ai-engineering/portfolio/regression-telemetry-gate
pip install -e . pytest
pytest -q                 # offline tests, no network/model needed
# run the API:
uvicorn regression_gate.app:app --port 8000
curl -s -X POST localhost:8000/gate -H 'content-type: application/json' -d '{
  "items":[{"id":"q1","answer":"connection pool exhausted","expected":"connection pool exhausted"}],
  "baseline":{"mean_score":0.77,"pass_rate":1.0}
}'
```

## Using in CI
Point your pipeline at `/gate` (or import `Gate`) with the current eval set and the
last known-good baseline. If `passed` is false, exit non-zero to block the merge.

## Layout
```
regression_gate/
  gate.py   # Gate, GateConfig, Metrics, compare, run_evals, default_scorer, emit_telemetry
  app.py    # FastAPI wrapper (/health /ready /gate)
tests/
  test_gate.py   # offline unit tests
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Note on the scorer
The default scorer is a deterministic token-F1 proxy so everything runs offline. For
real quality gating, pass an LLM-judge or RAGAS-style scorer — the gate logic is
unchanged.
