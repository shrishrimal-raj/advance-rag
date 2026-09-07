import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for Enterprise Search. Import-safe."""
from enterprise_search.search import SearchService

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

_DOCS = [
    ("d1", "error 4042 connection pool exhausted retry"),
    ("d2", "how to configure the database connection string"),
    ("d3", "user authentication and password reset flow"),
    ("d4", "connection timeout settings for external services"),
]


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="Enterprise Search Service")
    svc = SearchService(_DOCS)
    top_k = int(os.environ.get("SEARCH_TOP_K", "3"))

    class Q(BaseModel):
        query: str
        top_k: int = top_k

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "indexed": len(_DOCS)}

    @app.post("/search")
    def search(req: Q):
        return {"results": svc.search(req.query, top_k=req.top_k)}

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn enterprise_search.app:app --port 8000")
