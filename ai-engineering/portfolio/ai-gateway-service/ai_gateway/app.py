import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for the AI Gateway. Import-safe: builds only if fastapi present."""
from ai_gateway.gateway import AIGateway, GatewayConfig

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False


def _default_providers():
    def primary(prompt):
        if not os.environ.get("LLM_API_KEY"):
            raise RuntimeError("LLM_API_KEY not set")
        return f"(primary LLM) {prompt[:40]}"

    def fallback(prompt):
        return "(fallback echo) " + prompt[:40]

    return [primary, fallback]


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="AI Gateway Service")
    gw = AIGateway(_default_providers(), GatewayConfig())

    class Req(BaseModel):
        prompt: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        ok = bool(os.environ.get("LLM_API_KEY"))
        return {"ready": ok, "reason": None if ok else "LLM_API_KEY not set"}

    @app.post("/complete")
    def complete(req: Req):
        r = gw.complete(req.prompt)
        if not r.ok and r.degraded:
            raise HTTPException(status_code=503, detail=r.text)
        return {"text": r.text, "provider": r.provider, "attempts": r.attempts}

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn ai_gateway.app:app --port 8000")
