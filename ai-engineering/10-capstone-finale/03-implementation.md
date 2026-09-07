# 🔧 Week 10 Implementation — Build The Enterprise AI Platform

> Step-by-step. Each step ends with a **verify** line.

## 0. Setup
```bash
./.venv/bin/python -c "import langgraph; print('ok')"
```
**verify:** prints `ok`.

## 1. Production readiness (approach_2)
Validate config (fail fast), run health/readiness probes, demonstrate graceful degradation when a dependency is down, and emit a structured JSON log line. Assert each behavior.
**verify:** `python code/approach_2_production_readiness.py` passes all readiness checks and its self-check.

## 2. Monitoring & maintenance (approach_3)
Simulate a batch of runs with latencies/errors/answer-lengths; compute p50/p95 latency, error rate, and a drift proxy; alert on threshold breach. Assert detection of an injected anomaly.
**verify:** `python code/approach_3_monitoring_maintenance.py` detects the injected drift and passes its self-check.

## 3. Deployment strategies (benchmark)
Score batch vs streaming and on-prem vs cloud across criteria per scenario; recommend. Assert the right strategy wins each.
**verify:** `python code/benchmark_deployment_strategies.py` recommends correctly and passes its self-check.

## 4. The platform (main.py)
Wire the components: retrieve (local cosine RAG) -> agent (plan + synthesize, cloud LLM) -> evaluate (offline faithfulness/groundedness) -> trace (JSONL). `--selftest` runs one query end-to-end (≈2–3 LLM calls) and writes a trace file.
**verify:** `python code/main.py --selftest` returns a grounded answer, an eval score, and writes `trace.jsonl`.

## 5. Deploy artifacts
`code/app.py` (FastAPI `/health` + `/ask`), `Dockerfile`, `docker-compose.yml`, `.env.example`. Static - reviewed, not run here.
**verify:** `python -m py_compile code/app.py` is clean; Dockerfile/compose are well-formed.

## 6. Run everything
```bash
PY=./.venv/bin/python   # or .venv\Scripts\python.exe
$PY code/approach_2_production_readiness.py
$PY code/approach_3_monitoring_maintenance.py
$PY code/benchmark_deployment_strategies.py
$PY code/main.py --selftest
```

## Troubleshooting
- **Config invalid at startup** - fail fast with a clear message; never half-start.
- **Dependency down** - return a typed fallback response; log the degradation; don't 500.
- **Drift alert fires** - re-run the eval suite; diff against the last green baseline.
- **Container won't build** - pin the Python base image; copy only what's needed; keep secrets out of the image.
