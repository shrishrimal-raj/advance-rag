import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Dual-Agent Supervisor. Import-safe (builds only if fastapi present)."""
from dual_agent_supervisor.core import CheckerVerdict, DualAgentSupervisor, make_failover

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False


def _maker(t: str) -> str:
    return f"answer to: {t}"


def _checker(v: str) -> CheckerVerdict:
    return CheckerVerdict(approved=bool(v), reason="non-empty")


SUPERVISOR = DualAgentSupervisor(
    make_failover(_maker, _maker),
    make_failover(_checker, _checker),
    max_retries=3,
)


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Dual-Agent Supervisor")

    class RunReq(BaseModel):
        task: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "pattern": "maker-checker + failover"}

    @app.post("/run")
    def run(req: RunReq):
        return SUPERVISOR.run(req.task)

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn dual_agent_supervisor.app:app --port 8000")
