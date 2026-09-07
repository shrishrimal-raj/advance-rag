import os
import pathlib
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Thin FastAPI wrapper for the Enterprise AI Platform.

Exposes /health, /ready, and /ask. Import-safe: builds the app only if fastapi is
available, otherwise prints a hint. Run: uvicorn app:app --port 8000
"""
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
import main as plat  # noqa: E402

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:
    _HAVE_FASTAPI = False


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="Enterprise AI Platform")

    class AskRequest(BaseModel):
        query: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        ok = bool(os.environ.get("LLM_API_KEY"))
        return {"ready": ok, "reason": None if ok else "LLM_API_KEY not set"}

    @app.post("/ask")
    def ask(req: AskRequest):
        try:
            contexts = plat.retrieve(req.query)
            answer = plat.run_agent(req.query, contexts)
            ev = plat.evaluate(answer, contexts)
            return {"answer": answer, "eval": ev, "retrieved": [d["id"] for _, d in contexts]}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"degraded: {type(e).__name__}: {e}")

    return app


app = build_app()

if __name__ == "__main__":
    if app is None:
        print("fastapi not installed; install with: pip install fastapi uvicorn")
        raise SystemExit(0)
    print("Run with: uvicorn app:app --port 8000")
