# The Regression Gate

A **CI/CD eval harness** that runs evaluations on a golden dataset for every PR and
**blocks merges on regression**. Week 5's weekly build from the SDE → AI Engineer course,
packaged as a portfolio service.

## What it does
- Scores a golden dataset (question / answer / context / expected) with deterministic
  metrics: `answer_correctness`, `context_relevance`, `faithfulness`.
- Compares current scores against a stored **baseline**.
- Emits a PASS/FAIL verdict; any metric dropping more than the threshold is a regression
  that fails the gate (and thus blocks the merge in CI).

## Quick start
```bash
cd ai-engineering/portfolio/regression-gate
pip install -e . pytest
pytest -q                 # offline, no LLM / LangSmith needed
uvicorn regression_gate.app:app --port 8000
curl -X POST localhost:8000/gate -H 'content-type: application/json' \
  -d '{"cases":[{"question":"What is RAG?","answer":"RAG retrieves then generates.","context":"RAG retrieves documents then generates.","expected":"RAG retrieves then generates"}]}'
```

## Layout
```
regression_gate/
  core.py   # metrics, score_dataset, run_gate, GateResult, EvalCase
  app.py    # FastAPI wrapper (/health /ready /gate)
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Production notes
Metrics are heuristic so CI is green with no secrets or LLM calls. In production, swap the
metric functions for RAGAS scorers and push traces to LangSmith — the `run_gate` contract
(score → compare → verdict) is unchanged.
