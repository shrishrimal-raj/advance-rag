import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regression_gate.gate import Gate, GateConfig, Metrics, default_scorer, run_evals  # noqa: E402

GOOD = [
    {"id": "q1", "answer": "the connection pool is exhausted", "expected": "connection pool exhausted"},
    {"id": "q2", "answer": "reset the password now", "expected": "password reset"},
]
BAD = [
    {"id": "q1", "answer": "nothing relevant at all", "expected": "connection pool exhausted"},
    {"id": "q2", "answer": "completely different text", "expected": "password reset"},
]
BASELINE_GOOD = Metrics(n=2, mean_score=0.70, pass_rate=1.0)


def test_default_scorer_bounds():
    assert default_scorer("connection pool exhausted", "connection pool exhausted") == 1.0
    assert default_scorer("zzz qqq", "connection pool exhausted") == 0.0


def test_healthy_passes():
    g = Gate(GateConfig())
    v = g.run(GOOD, baseline=BASELINE_GOOD)
    assert v.passed is True
    assert v.reasons == []


def test_regression_detected():
    g = Gate(GateConfig())
    v = g.run(BAD, baseline=BASELINE_GOOD)
    assert v.passed is False
    assert any("regression" in r for r in v.reasons)


def test_low_pass_rate_without_baseline():
    g = Gate(GateConfig())
    v = g.run(BAD, baseline=None)
    assert v.passed is False
    assert any("pass rate" in r for r in v.reasons)


def test_run_evals_aggregates():
    _, m = run_evals(GOOD, default_scorer, 0.6)
    assert m.n == 2
    assert m.pass_rate >= 0.8


def test_telemetry_written(tmp_path):
    p = tmp_path / "telemetry.jsonl"
    g = Gate(GateConfig())
    g.run(GOOD, baseline=BASELINE_GOOD, telemetry_path=str(p))
    line = p.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["passed"] is True
    assert data["gate"] == "regression-telemetry"
