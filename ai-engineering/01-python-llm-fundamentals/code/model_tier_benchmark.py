import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Model tier benchmark - head-to-head latency/cost across model tiers.

Measures TTFT, total latency, estimated tokens, and estimated cost for identical
prompts across tiers. Demonstrates the tiering trade-off from the learning doc.

8GB note: makes a handful of cloud calls. Use --quick for fewer iterations.
Set BENCH_SMALL_MODEL / BENCH_LARGE_MODEL to compare genuinely different models.
"""
import argparse
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
KEY = os.getenv("YOLO_AUTO_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
# (label, model, illustrative $/1M prompt, $/1M completion)
TIERS = [
    ("small", os.getenv("BENCH_SMALL_MODEL", "qwen3.8-27b"), 0.30, 0.90),
    ("large", os.getenv("BENCH_LARGE_MODEL", "qwen3.8-27b"), 1.50, 4.50),
]
PROMPT = "In one sentence, what is retrieval-augmented generation?"


async def one(client, model: str):
    t0 = time.time()
    first = None
    comp_chars = 0
    headers = {"Authorization": f"Bearer {KEY}"}
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "stream": True}
    async with client.stream("POST", f"{BASE_URL}/chat/completions",
                             json=body, headers=headers) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            p = line[len("data:"):].strip()
            if p == "[DONE]":
                break
            try:
                o = json.loads(p)
            except Exception:
                continue
            d = o.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if d:
                if first is None:
                    first = time.time() - t0
                comp_chars += len(d)
    ttft = first if first is not None else (time.time() - t0)
    total = time.time() - t0
    pt = max(1, len(PROMPT) // 4)
    ct = max(1, comp_chars // 4)
    return ttft, total, pt, ct


async def run(iters: int) -> None:
    if not KEY:
        print("[no key] set YOLO_AUTO_API_KEY to run the benchmark")
        return
    import httpx

    rows = []
    async with httpx.AsyncClient(timeout=60) as client:
        for label, model, pp, cp in TIERS:
            acc = [0.0, 0.0, 0.0, 0.0]
            ok = 0
            for _ in range(iters):
                try:
                    ttft, total, pt, ct = await one(client, model)
                    acc[0] += ttft
                    acc[1] += total
                    acc[2] += pt
                    acc[3] += ct
                    ok += 1
                except Exception as e:
                    print(f"[warn] {label} iter failed: {e}")
            n = max(1, ok)
            avg_ttft, avg_total, avg_pt, avg_ct = acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n
            cost = (avg_pt * pp + avg_ct * cp) / 1_000_000
            rows.append((label, model, avg_ttft, avg_total, avg_pt, avg_ct, cost))

    print()
    hdr = "tier     model              TTFT(s)  total(s)   ptok   ctok   cost$"
    print(hdr)
    for r in rows:
        print(f"{r[0]:<8}{r[1]:<18}{r[2]:>9.2f}{r[3]:>10.2f}{r[4]:>7.0f}{r[5]:>7.0f}{r[6]:>9.4f}")
    print()
    print("(pricing illustrative; latency measured live)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Model tier benchmark")
    ap.add_argument("--quick", action="store_true", help="fewer iterations (lighter on laptop)")
    a = ap.parse_args()
    asyncio.run(run(2 if a.quick else 3))
