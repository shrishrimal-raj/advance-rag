import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Capstone. Import-safe (builds only if fastapi present).

Exposes a non-streaming `/ask` and a Server-Sent-Events `/ask/stream` for the live UI.
In production, wrap the graph with LangSmith tracing (LANGSMITH_TRACING=true).
"""
from capstone.core import build_capstone_graph

try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

GRAPH = build_capstone_graph(lambda q: f"context for {q}")


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Capstone")

    class AskReq(BaseModel):
        query: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "stack": "multi-agent + RAG + streaming"}

    @app.post("/ask")
    def ask(req: AskReq):
        st = GRAPH.run(req.query)
        return {"final": st.final, "events": st.events}

    @app.post("/ask/stream")
    def ask_stream(req: AskReq):
        def gen():
            for event, state in GRAPH.stream(req.query):
                yield f"data: {json.dumps({'event': event, 'final': state.final})}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn capstone.app:app --port 8000")
