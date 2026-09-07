import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Knowledge Engine. Import-safe (builds only if fastapi present)."""
from knowledge_engine.core import Chunk, KnowledgeEngine

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

# Seed corpus (production loads from pgvector).
ENGINE = KnowledgeEngine()
ENGINE.index([
    Chunk("c1", "PostgreSQL indexes speed up queries", "db-guide", {"topic": "db"}, [1, 0, 0]),
    Chunk("c2", "Redis caching reduces database load", "cache-guide", {"topic": "db"}, [0, 1, 0]),
    Chunk("c3", "React components render UI", "fe-guide", {"topic": "fe"}, [0, 0, 1]),
])


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Knowledge Engine")

    class SearchReq(BaseModel):
        query: str
        top_k: int = 5
        filters: dict = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "chunks": len(ENGINE.chunks)}

    @app.post("/search")
    def search(req: SearchReq):
        res = ENGINE.retrieve(req.query, top_k=req.top_k, filters=req.filters)
        return {"results": [{"id": r["chunk"].id, "text": r["chunk"].text,
                             "source": r["chunk"].source, "score": r["score"],
                             "citation": r["citation"]} for r in res]}

    @app.post("/answer")
    def answer(req: SearchReq):
        return ENGINE.answer_context(req.query, top_k=req.top_k, filters=req.filters)

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn knowledge_engine.app:app --port 8000")
