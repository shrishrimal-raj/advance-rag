"""FastAPI entrypoint for the Agentic Research Agent.

Run:  uv run uvicorn main:app --port 8000
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI

from agent.tools import get_tools
from api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Research Agent",
        description="Plan-and-execute + reflection RAG agent with streaming API",
        version="0.1.0",
    )
    app.state.llm = None  # set lazily; run_research falls back to its own default LLM
    try:
        from agent.graph import _default_llm
        app.state.llm = _default_llm()
    except Exception:
        pass  # no key configured — /health reports 'missing', /ask degrades with a clear error
    try:
        app.state.tools = get_tools()
    except Exception:
        app.state.tools = None
    app.include_router(router)
    return app


app = create_app()
