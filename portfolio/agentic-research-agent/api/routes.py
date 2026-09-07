"""API routes for the agentic research agent."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.graph import run_research, stream_research

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def _deps(request: Any):
    """LLM/tools are attached to the app by create_app() for testability."""
    app = request.app
    return app.state.llm, app.state.tools


@router.get("/health")
def health(request: Any) -> dict[str, Any]:
    _, tools = _deps(request)
    try:
        corpus_size = tools.collection.count()
    except Exception:  # noqa: BLE001
        corpus_size = -1
    return {
        "status": "ok",
        "llm": "configured" if request.app.state.llm is not None else "missing",
        "corpus_docs": corpus_size,
    }


@router.post("/ask")
async def ask_impl(payload: AskRequest, request: Any) -> dict[str, Any]:
    llm, tools = _deps(request)
    try:
        state = run_research(payload.question, llm=llm, tools=tools)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"research run failed: {exc}. Check YOLO_AUTO_API_KEY in .env and that the corpus is ingested (scripts/ingest.py).",
        ) from exc
    return {
        "question": payload.question,
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "plan": state.get("plan", []),
        "reflections": state.get("reflections", []),
    }


@router.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Any) -> StreamingResponse:
    llm, tools = _deps(request)

    def event_stream():
        try:
            for kind, data in stream_research(payload.question, llm=llm, tools=tools):
                yield f"data: {json.dumps({'type': kind, 'payload': data})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'payload': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
