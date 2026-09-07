"""Regression Telemetry Gate core.

Runs evals over a dataset with a pluggable scorer, aggregates metrics, compares them
against a stored baseline, detects regression, and emits structured telemetry. The
default scorer is a deterministic token-F1 proxy so the whole gate runs offline;
swap in an LLM-judge or RAGAS-style scorer by passing your own `scorer(answer,
expected) -> float`.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def default_scorer(answer: str, expected: str) -> float:
    """Deterministic offline proxy: token-set F1 between answer and expected."""
    a = set(_tokenize(answer))
    e = set(_tokenize(expected))
    if not e:
        return 0.0
    inter = len(a & e)
    prec = inter / len(a) if a else 0.0
    rec = inter / len(e)
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0


@dataclass
class GateConfig:
    pass_threshold: float = 0.6
    regression_threshold: float = 0.05
    min_pass_rate: float = 0.8


@dataclass
class ItemResult:
    id: str
    score: float
    passed: bool


@dataclass
class Metrics:
    n: int
    mean_score: float
    pass_rate: float


@dataclass
class GateVerdict:
    passed: bool
    reasons: List[str]
    current: Metrics
    baseline: Optional[Metrics]
    delta_mean: Optional[float]
    telemetry: dict = field(default_factory=dict)


def run_evals(items: List[dict], scorer: Callable[[str, str], float] = default_scorer,
              pass_threshold: float = 0.6):
    results: List[ItemResult] = []
    for it in items:
        s = max(0.0, min(1.0, float(scorer(it["answer"], it["expected"]))))
        results.append(ItemResult(id=it["id"], score=s, passed=s >= pass_threshold))
    n = len(results)
    mean = sum(r.score for r in results) / n if n else 0.0
    prate = sum(1 for r in results if r.passed) / n if n else 0.0
    return results, Metrics(n=n, mean_score=round(mean, 4), pass_rate=round(prate, 4))


def compare(current: Metrics, baseline: Optional[Metrics], cfg: GateConfig) -> GateVerdict:
    reasons: List[str] = []
    if current.pass_rate < cfg.min_pass_rate:
        reasons.append(f"pass rate {current.pass_rate:.2f} < {cfg.min_pass_rate:.2f}")
    delta = None
    if baseline is not None:
        delta = round(current.mean_score - baseline.mean_score, 4)
        if delta < -cfg.regression_threshold:
            reasons.append(
                f"regression: mean {current.mean_score:.3f} dropped {abs(delta):.3f} "
                f"from baseline {baseline.mean_score:.3f} (> {cfg.regression_threshold})"
            )
    passed = not reasons
    telemetry = {
        "gate": "regression-telemetry",
        "passed": passed,
        "current": {"n": current.n, "mean_score": current.mean_score, "pass_rate": current.pass_rate},
        "baseline": None if baseline is None else {"mean_score": baseline.mean_score, "pass_rate": baseline.pass_rate},
        "delta_mean": delta,
        "reasons": reasons,
    }
    return GateVerdict(passed=passed, reasons=reasons, current=current, baseline=baseline,
                       delta_mean=delta, telemetry=telemetry)


def emit_telemetry(telemetry: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(telemetry) + "\n")


class Gate:
    def __init__(self, config: Optional[GateConfig] = None, scorer: Callable[[str, str], float] = default_scorer):
        self.config = config or GateConfig()
        self.scorer = scorer

    def run(self, items: List[dict], baseline: Optional[Metrics] = None, telemetry_path: Optional[str] = None) -> GateVerdict:
        _, metrics = run_evals(items, self.scorer, self.config.pass_threshold)
        verdict = compare(metrics, baseline, self.config)
        if telemetry_path:
            emit_telemetry(verdict.telemetry, telemetry_path)
        return verdict
