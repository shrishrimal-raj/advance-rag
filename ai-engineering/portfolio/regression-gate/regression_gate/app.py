import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Regression Gate. Import-safe (builds only if fastapi present)."""
from regression_gate.core import EvalCase, METRICS, run_gate

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

# Default golden baseline (production stores this as an artifact from the last green merge).
DEFAULT_BASELINE = {m: 1.0 for m in METRICS}


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Regression Gate")

    class CaseIn(BaseModel):
        question: str
        answer: str
        context: str
        expected: str

    class GateReq(BaseModel):
        cases: list
        baseline: dict = None
        threshold: float = 0.05

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "metrics": list(METRICS)}

    @app.post("/gate")
    def gate(req: GateReq):
        cases = [EvalCase(c.question, c.answer, c.context, c.expected) for c in req.cases]
        base = req.baseline or DEFAULT_BASELINE
        return run_gate(cases, base, req.threshold).as_dict()

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn regression_gate.app:app --port 8000")
