import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Multimodal RAG Engine. Import-safe (builds only if fastapi present).

In production the processors are backed by CLIP (image), Whisper (audio), and a text
embedder; here they are lightweight stand-ins so the service runs and tests offline.
"""
from multimodal_rag_engine.core import MultimodalRAGEngine

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

ENGINE = MultimodalRAGEngine({
    "text": lambda it: it.get("text", "").strip(),
    "image": lambda it: f"image:{it.get('src', '')}",
    "audio": lambda it: f"transcript:{it.get('wav', '')}",
})


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Multimodal RAG Engine")

    class ItemIn(BaseModel):
        type: str = "text"
        text: str = ""
        src: str = ""
        wav: str = ""

    class QueryReq(BaseModel):
        items: list

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "modalities": ["text", "image", "audio"]}

    @app.post("/query")
    def query(req: QueryReq):
        items = [i.model_dump() if hasattr(i, "model_dump") else i.dict() for i in req.items]
        ctx = ENGINE.fuse(items)
        return {"context": ctx, "dashboard": ENGINE.dashboard()}

    @app.get("/dashboard")
    def dash():
        return ENGINE.dashboard()

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn multimodal_rag_engine.app:app --port 8000")
