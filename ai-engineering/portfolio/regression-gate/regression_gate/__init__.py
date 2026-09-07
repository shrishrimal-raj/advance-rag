"""The Regression Gate: CI eval harness that blocks merges on metric regression."""
from .core import EvalCase, GateResult, METRICS, run_gate, score_dataset

__all__ = ["EvalCase", "GateResult", "METRICS", "run_gate", "score_dataset"]
