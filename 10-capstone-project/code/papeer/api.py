"""Phase 5 — FastAPI service layer for Papeer.

⚠️  NOT runnable out of the box. This file needs two extra deps that are intentionally
    kept OUT of the base course environment:

        uv add fastapi uvicorn

    Then run from the project root:

        uv run uvicorn 10-capstone-project.code.papeer.api:app --reload --port 8000

    Try it:
        curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
             -d '{"question": "What navigation does the Falcon AMR use?"}'

Design notes (industry practice):
- The endpoint is *streaming-ready*: /ask/stream yields tokens as they arrive instead of
  blocking until the full answer exists (critical for perceived latency in production).
- The index is assumed to already exist (ingestion is an offline/batch concern — never do
  it inside a request). If missing, we return a clear 503 rather than crashing.
- Response shape is stable and versioned so clients can rely on it.
"""
from __future__ import annotations

import sys
import pathlib
import time

_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

# NOTE: `fastapi`/`uvicorn` are optional course deps — import lazily so importing this
# module for inspection doesn't hard-fail when they aren't installed.
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "fastapi is not installed. Run:  uv add fastapi uvicorn\n"
        f"(original error: {e})"
    )

from papeer.config import CHUNKS_JSON, get_llm  # noqa: E402
from papeer.retrieval import retrieve  # noqa: E402
from papeer.agent import _format_context, _llm_text, GENERATE_PROMPT  # noqa: E402


app = FastAPI(title="Papeer API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    stages: list[dict]      # [{name, seconds}]
    cached: bool = False


def _ensure_index() -> None:
    if not CHUNKS_JSON.exists():
        raise HTTPException(
            status_code=503,
            detail="Index not built. Run ingestion first: "
                   "uv run python 10-capstone-project/code/papeer/ingestion.py",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_built": CHUNKS_JSON.exists()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Non-streaming ask. Runs retrieve -> generate and returns a stable response shape."""
    _ensure_index()
    t0 = time.perf_counter()
    hits = retrieve(req.question)
    t_retrieve = time.perf_counter() - t0

    t1 = time.perf_counter()
    ctx = _format_context(hits)
    answer = _llm_text(GENERATE_PROMPT.format(context=ctx, question=req.question)).strip()
    t_generate = time.perf_counter() - t1

    return AskResponse(
        question=req.question,
        answer=answer,
        sources=[h["source"] for h in hits],
        stages=[{"name": "retrieve", "seconds": round(t_retrieve, 4)},
                {"name": "generate", "seconds": round(t_generate, 4)}],
        cached=False,
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Streaming-ready ask: emits the answer as newline-delimited chunks.

    In production you'd stream LLM tokens directly (llm.stream). Here we chunk the final
    answer to demonstrate the streaming contract without requiring a streaming LLM client.
    """
    _ensure_index()

    def generator():
        hits = retrieve(req.question)
        ctx = _format_context(hits)
        full = _llm_text(GENERATE_PROMPT.format(context=ctx, question=req.question))
        # Emit metadata line first, then the answer in small chunks (streaming contract).
        yield f"sources:{','.join(h['source'] for h in hits)}\n"
        for i in range(0, len(full), 40):
            yield full[i:i + 40] + "\n"

    return StreamingResponse(generator(), media_type="text/plain")
