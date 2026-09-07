import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The AI Gateway. Import-safe (builds only if fastapi present)."""
from ai_gateway.core import MODEL_PRICING, TokenCostTracker, stream_chat

try:
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

TRACKER = TokenCostTracker()


def _default_provider(prompt: str):
    async def gen():
        # Stub: production wraps an OpenAI/OpenRouter streaming client here.
        yield f"(echo) {prompt[:80]}"

    return gen()


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The AI Gateway")

    class ChatReq(BaseModel):
        prompt: str
        model: str = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": bool(os.environ.get("LLM_API_KEY")), "models": list(MODEL_PRICING)}

    @app.get("/usage")
    def usage():
        return TRACKER.summary()

    @app.post("/chat/stream")
    async def chat_stream(req: ChatReq):
        async def gen():
            async for ev in stream_chat(req.prompt, req.model, _default_provider(req.prompt), TRACKER):
                yield f"data: {json.dumps(ev)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn ai_gateway.app:app --port 8000")
