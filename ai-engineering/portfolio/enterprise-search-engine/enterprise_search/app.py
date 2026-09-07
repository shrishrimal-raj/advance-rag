import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Enterprise Search Engine. Import-safe (builds only if fastapi present)."""
from enterprise_search.core import Doc, EnterpriseSearchEngine

try:
    from fastapi import FastAPI, Header
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

# Seed namespace (production loads per-tenant from Pinecone).
ENGINE = EnterpriseSearchEngine()
ENGINE.upsert("demo", [
    Doc("d1", "Getting started with the platform", "docs", {"type": "guide"}, [1, 0]),
    Doc("d2", "API rate limits explained", "docs", {"type": "guide"}, [0, 1]),
])


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Enterprise Search Engine")

    class SearchReq(BaseModel):
        query: str
        top_k: int = 5
        filters: dict = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "tenants": ENGINE.tenants()}

    @app.post("/search")
    def search(req: SearchReq, x_tenant_id: str = Header(default="demo")):
        res = ENGINE.search(x_tenant_id, req.query, top_k=req.top_k, filters=req.filters)
        return {"tenant": x_tenant_id,
                "results": [{"id": r["doc"].id, "text": r["doc"].text,
                             "source": r["doc"].source, "score": r["score"],
                             "citation": r["citation"]} for r in res]}

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn enterprise_search.app:app --port 8000")
