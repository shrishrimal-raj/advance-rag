import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for the Regression Telemetry Gate. Import-safe."""
from regression_gate.gate import Gate, GateConfig, Metrics

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False


def _config_from_env():
    return GateConfig(
        pass_threshold=float(os.environ.get("GATE_PASS_THRESHOLD", "0.6")),
        regression_threshold=float(os.environ.get("GATE_REGRESSION_THRESHOLD", "0.05")),
        min_pass_rate=float(os.environ.get("GATE_MIN_PASS_RATE", "0.8")),
    )


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="Regression Telemetry Gate")
    gate = Gate(_config_from_env())

    class Item(BaseModel):
        id: str
        answer: str
        expected: str

    class Baseline(BaseModel):
        mean_score: float
        pass_rate: float

    class GateReq(BaseModel):
        items: list
        baseline: Baseline = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True}

    @app.post("/gate")
    def gate_endpoint(req: GateReq):
        baseline = None
        if req.baseline is not None:
            baseline = Metrics(n=len(req.items), mean_score=req.baseline.mean_score, pass_rate=req.baseline.pass_rate)
        verdict = gate.run(req.items, baseline=baseline)
        return verdict.telemetry

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn regression_gate.app:app --port 8000")
