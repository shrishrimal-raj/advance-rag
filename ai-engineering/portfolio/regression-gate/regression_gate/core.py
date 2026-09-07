"""The Regression Gate core.

A CI eval harness that scores a golden dataset with deterministic metrics, compares
against a baseline, and emits a PASS/FAIL verdict that blocks merges on regression.
Metrics are heuristic (no LLM) so the gate is fully offline-testable; swap in
RAGAS/LangSmith scorers in production without changing the gate contract.
"""
from dataclasses import dataclass, field
from typing import Dict, List


def _tokens(s: str) -> List[str]:
    return [t for t in s.lower().replace("\n", " ").split() if t]


def answer_correctness(answer: str, expected: str) -> float:
    """Fraction of expected's key tokens present in the answer (0..1)."""
    et = set(_tokens(expected))
    if not et:
        return 1.0
    at = set(_tokens(answer))
    return len(et & at) / len(et)


def context_relevance(context: str, question: str) -> float:
    """Fraction of question tokens found in the context (0..1)."""
    qt = set(_tokens(question))
    if not qt:
        return 1.0
    ct = set(_tokens(context))
    return len(qt & ct) / len(qt)


def faithfulness(answer: str, context: str) -> float:
    """Proxy: fraction of answer tokens grounded in the context (0..1)."""
    at = set(_tokens(answer))
    if not at:
        return 1.0
    ct = set(_tokens(context))
    return len(at & ct) / len(at)


METRICS = {
    "answer_correctness": lambda q, a, c, e: answer_correctness(a, e),
    "context_relevance": lambda q, a, c, e: context_relevance(c, q),
    "faithfulness": lambda q, a, c, e: faithfulness(a, c),
}


@dataclass
class EvalCase:
    question: str
    answer: str
    context: str
    expected: str


def score_dataset(cases: List[EvalCase]) -> Dict[str, float]:
    """Mean of each metric across all cases."""
    acc = {m: [] for m in METRICS}
    for c in cases:
        for m, fn in METRICS.items():
            acc[m].append(fn(c.question, c.answer, c.context, c.expected))
    return {m: (sum(v) / len(v) if v else 0.0) for m, v in acc.items()}


@dataclass
class GateResult:
    passed: bool
    current: Dict[str, float]
    baseline: Dict[str, float]
    regressions: List[Dict] = field(default_factory=list)

    def as_dict(self):
        return {"passed": self.passed, "current": self.current,
                "baseline": self.baseline, "regressions": self.regressions}


def run_gate(cases: List[EvalCase], baseline: Dict[str, float],
             threshold: float = 0.05) -> GateResult:
    """Score `cases`, compare each metric to `baseline`; FAIL if any drops > threshold."""
    current = score_dataset(cases)
    regressions = []
    for m in METRICS:
        cur = current.get(m, 0.0)
        base = baseline.get(m, cur)
        delta = cur - base
        if delta < -threshold:
            regressions.append({"metric": m, "baseline": round(base, 4),
                                "current": round(cur, 4), "delta": round(delta, 4)})
    return GateResult(
        passed=len(regressions) == 0,
        current={k: round(v, 4) for k, v in current.items()},
        baseline={k: round(v, 4) for k, v in baseline.items()},
        regressions=regressions,
    )
