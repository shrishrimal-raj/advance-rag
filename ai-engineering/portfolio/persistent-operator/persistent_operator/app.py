import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Persistent Operator. Import-safe (builds only if fastapi present).

Uses an in-memory store by default; point REDIS_URL at Redis and swap MemoryStore for a
redis-backed store in production.
"""
from persistent_operator.core import MemoryStore, PersistentOperator

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

STORE = MemoryStore()
TOOLS = {
    "echo": lambda text: text,
    "upper": lambda text: text.upper(),
}


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Persistent Operator")

    class MsgReq(BaseModel):
        message: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "tools": list(TOOLS)}

    @app.post("/sessions/{sid}/message")
    def post_message(sid: str, req: MsgReq):
        op = PersistentOperator(STORE, sid, TOOLS)
        return op.step(req.message, lambda st: {"answer": f"echo: {req.message}"})

    @app.get("/sessions/{sid}/history")
    def get_history(sid: str):
        return PersistentOperator(STORE, sid, TOOLS).history()

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn persistent_operator.app:app --port 8000")
