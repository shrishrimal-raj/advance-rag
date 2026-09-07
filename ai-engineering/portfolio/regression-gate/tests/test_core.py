import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regression_gate.core import (  # noqa: E402
    EvalCase,
    METRICS,
    answer_correctness,
    context_relevance,
    faithfulness,
    run_gate,
    score_dataset,
)

CASES_GOOD = [
    EvalCase("What is RAG?", "RAG retrieves then generates.",
             "RAG retrieves documents then generates.", "RAG retrieves then generates"),
    EvalCase("What is BM25?", "BM25 is a ranking function.",
             "BM25 ranks by term frequency.", "BM25 is a ranking function"),
]


def test_metrics_bounded_and_zero_on_mismatch():
    assert 0.0 <= answer_correctness("a b", "a b") <= 1.0
    assert answer_correctness("totally different", "a b") == 0.0
    assert 0.0 <= context_relevance("q", "context") <= 1.0
    assert 0.0 <= faithfulness("a", "c") <= 1.0


def test_score_dataset_shape_and_bounds():
    s = score_dataset(CASES_GOOD)
    assert set(s.keys()) == set(METRICS.keys())
    assert all(0.0 <= v <= 1.0 for v in s.values())


def test_gate_passes_when_no_regression():
    base = score_dataset(CASES_GOOD)
    res = run_gate(CASES_GOOD, base, threshold=0.05)
    assert res.passed is True
    assert res.regressions == []


def test_gate_fails_on_regression():
    base = score_dataset(CASES_GOOD)
    bad = [EvalCase(c.question, "irrelevant noise", c.context, c.expected) for c in CASES_GOOD]
    res = run_gate(bad, base, threshold=0.05)
    assert res.passed is False
    assert any(r["metric"] == "answer_correctness" for r in res.regressions)


def test_gate_result_serializable():
    base = score_dataset(CASES_GOOD)
    d = run_gate(CASES_GOOD, base).as_dict()
    assert d["passed"] is True
    assert "current" in d and "baseline" in d
