import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 2 - raw OpenAI-compatible streaming client (no LangChain, no FastAPI).

Proves you understand the wire protocol: POST /chat/completions with stream=true,
then manually parse Server-Sent Events (data: {...}, terminal data: [DONE]).
"""
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv  # noqa: E402

# Load repo-root .env (holds the working YOLO_AUTO_API_KEY), then any course-local .env.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BASE_URL = os.getenv("YOLO_AUTO_BASE_URL", "https://yolo-auto.com/v1").rstrip("/")
MODEL = os.getenv("YOLO_AUTO_MODEL", "qwen3.8-27b")
KEY = os.getenv("YOLO_AUTO_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
PROMPT = "Explain the attention mechanism in one sentence."


async def stream() -> None:
    if not KEY:
        print("[no key] set YOLO_AUTO_API_KEY (see .env.example) or copy ../.env")
        return
    import httpx

    headers = {"Authorization": f"Bearer {KEY}"}
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "stream": True}
    t0 = time.time()
    first = None
    comp_chars = 0
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", f"{BASE_URL}/chat/completions",
                                 json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    if first is None:
                        first = time.time() - t0
                    comp_chars += len(delta)
                    print(delta, end="", flush=True)
    print()
    ttft = first if first is not None else (time.time() - t0)
    print(f"[raw] model={MODEL} TTFT={ttft:.2f}s completion~{comp_chars // 4}tok "
          f"total={time.time() - t0:.2f}s")


if __name__ == "__main__":
    asyncio.run(stream())
