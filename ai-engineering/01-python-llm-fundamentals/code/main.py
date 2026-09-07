import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The AI Gateway (Week 1 weekly build).

A streaming, non-blocking chat backend with real-time token usage and cost tracking.
Framework-based approach: FastAPI + SSE + LangChain OpenAI-compatible client.

Run the server:
    uv run uvicorn main:create_app --factory --app-dir <this-dir> --port 8000
Quick self-test (no server, one streaming call):
    uv run python main.py --selftest
"""
import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

sys.path.append(str(Path(__file__).resolve().parents[2]))
from shared.config import get_llm, get_llm_provider_name  # noqa: E402

NL = chr(10)

# Illustrative $/1M-token rates. Override per deployment; this is a teaching default.
DEFAULT_PRICING = {"prompt": 0.50, "completion": 1.50}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, pricing: dict = DEFAULT_PRICING) -> float:
        return (self.prompt_tokens * pricing["prompt"]
                + self.completion_tokens * pricing["completion"]) / 1_000_000


def est_tokens(text: str) -> int:
    # Cheap heuristic (~4 chars/token). Real usage is used when the provider supplies it.
    return max(1, len(text) // 4)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}" + NL + NL


def _sse_done() -> str:
    return "data: [DONE]" + NL + NL


def _have_key() -> bool:
    return bool(os.getenv("YOLO_AUTO_API_KEY", "").strip()
                or os.getenv("OPENAI_API_KEY", "").strip())


async def stream_chat(message: str) -> AsyncIterator[str]:
    """Yield SSE events: one per token, then a final usage/cost event, then [DONE]."""
    if not _have_key():
        yield _sse({"error": "no LLM key configured",
                    "hint": "set YOLO_AUTO_API_KEY (see .env.example) or copy ../.env"})
        yield _sse_done()
        return

    llm = get_llm(temperature=0.0)
    usage = Usage(prompt_tokens=est_tokens(message))
    parts: list[str] = []
    try:
        async for chunk in llm.astream([{"role": "user", "content": message}]):
            piece = getattr(chunk, "content", "") or ""
            if piece:
                parts.append(piece)
                usage.completion_tokens += est_tokens(piece)
                yield _sse({"token": piece})
    except Exception as e:  # graceful degradation: never hang the client
        yield _sse({"error": f"stream failed: {e}"})
        yield _sse_done()
        return

    ev = {
        "answer": "".join(parts),
        "usage": {"prompt_tokens": usage.prompt_tokens,
                  "completion_tokens": usage.completion_tokens,
                  "total_tokens": usage.total()},
        "cost_usd": round(usage.cost(), 6),
        "provider": get_llm_provider_name(),
    }
    yield _sse(ev)
    yield _sse_done()


def create_app():
    """App factory: FastAPI imported lazily so the module stays importable without a server."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="The AI Gateway", version="1.0")

    class ChatReq(BaseModel):
        message: str

    @app.get("/health")
    async def health():
        return {"status": "ok", "provider": get_llm_provider_name()}

    @app.post("/chat")
    async def chat(req: ChatReq):
        return StreamingResponse(stream_chat(req.message), media_type="text/event-stream")

    return app


async def _selftest(message: str) -> None:
    print(f"[selftest] provider={get_llm_provider_name()}")
    got_token = False
    async for ev in stream_chat(message):
        line = ev.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if "token" in obj:
            print(obj["token"], end="", flush=True)
            got_token = True
        elif "error" in obj:
            print()
            print("[selftest] ERROR:", obj.get("hint", obj["error"]))
        elif "usage" in obj:
            u = obj["usage"]
            print()
            print("[selftest] tokens=" + str(u["total_tokens"]) + " cost=$" + str(obj["cost_usd"]))
    if not got_token:
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The AI Gateway")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--message", default="Say hello in exactly three words.")
    a = ap.parse_args()
    if a.selftest:
        asyncio.run(_selftest(a.message))
    else:
        print("Start the server with:")
        print("  uv run uvicorn main:create_app --factory --app-dir <this-dir> --port 8000")
