# DEPLOYMENT — Regression Telemetry Gate

## Local
```bash
pip install -e .
uvicorn regression_gate.app:app --port 8000
curl -s localhost:8000/health
```

## Docker
```bash
docker build -t regression-gate .
docker run -p 8000:8000 regression-gate
```

## Compose
```bash
cp .env.example .env
docker compose up -d
```

## CI integration
`.github/workflows/ci.yml` runs the offline test suite. To use the gate itself as a
build blocker, add a step that calls `/gate` with the current eval set and the last
known-good baseline, then exits non-zero when `passed` is false:

```bash
RESP=$(curl -s -X POST localhost:8000/gate -H 'content-type: application/json' -d @eval_payload.json)
echo "$RESP"
echo "$RESP" | python -c "import sys,json; sys.exit(0 if json.load(sys.stdin)['passed'] else 1)"
```

## Rollout checklist
- [ ] Set thresholds in `.env` to match your quality bar.
- [ ] Store a known-good baseline before enabling the gate.
- [ ] Confirm `/ready` 200 and `/gate` returns a telemetry record.
- [ ] Verify a deliberately regressed payload fails the gate.
