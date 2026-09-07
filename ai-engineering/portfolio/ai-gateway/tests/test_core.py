import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_gateway.core import (  # noqa: E402
    TokenCostTracker,
    compute_cost,
    estimate_tokens,
    stream_chat,
)


def _provider(*parts):
    async def prov(prompt):
        for p in parts:
            yield p

    return prov


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10


def test_compute_cost_known_model():
    assert abs(compute_cost("gpt-4o-mini", 1_000_000, 0) - 0.15) < 1e-9


def test_compute_cost_unknown_model_uses_default():
    assert abs(compute_cost("mystery-model", 1_000_000, 1_000_000) - 4.0) < 1e-9


def test_stream_yields_tokens_then_usage():
    async def run():
        return [e async for e in stream_chat("hello world", "gpt-4o-mini", _provider("Hi", " there"))]

    evs = asyncio.run(run())
    assert [e["type"] for e in evs] == ["token", "token", "usage"]
    assert evs[-1]["output_tokens"] >= 1
    assert evs[-1]["cost_usd"] > 0


def test_tracker_aggregates():
    async def run():
        t = TokenCostTracker()
        for _ in range(3):
            [e async for e in stream_chat("hi", "gpt-4o-mini", _provider("ok"), t)]
        return t.summary()

    s = asyncio.run(run())
    assert s["requests"] == 3
    assert s["cost_usd"] > 0
    assert s["total_tokens"] == s["input_tokens"] + s["output_tokens"]
