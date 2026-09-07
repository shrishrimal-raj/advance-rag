"""The AI Gateway core.

A streaming, non-blocking chat backend with real-time token usage and cost
tracking. Providers are async callables `prompt -> AsyncIterator[str]` so the whole
thing is testable offline with a fake provider. In production the default provider
wraps an OpenAI/OpenRouter streaming client.
"""
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

# Per-million-token prices (USD). Extend as your catalog grows.
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}
DEFAULT_PRICING = {"input": 1.00, "output": 3.00}


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic (offline, no tokenizer dependency)."""
    return max(1, len(text) // 4)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return round((input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000, 8)


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def as_dict(self):
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": round(self.latency_ms, 2),
        }


class TokenCostTracker:
    """Aggregates usage/cost across many requests (real-time accounting)."""

    def __init__(self):
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def record(self, u: Usage):
        self.requests += 1
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cost_usd += u.cost_usd

    def summary(self):
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


async def stream_chat(
    prompt: str,
    model: str,
    provider: Callable[[str], AsyncIterator[str]],
    tracker: Optional[TokenCostTracker] = None,
) -> AsyncIterator[dict]:
    """Stream chunks from a provider as SSE-style events, ending with a usage event.

    Yields `{"type":"token","content":...}` per chunk, then one
    `{"type":"usage", ...}` with token counts, cost, and latency.
    """
    start = time.monotonic()
    parts = []
    async for chunk in provider(prompt):
        parts.append(chunk)
        yield {"type": "token", "content": chunk}
    text = "".join(parts)
    in_tok = estimate_tokens(prompt)
    out_tok = estimate_tokens(text)
    usage = Usage(
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=compute_cost(model, in_tok, out_tok),
        latency_ms=(time.monotonic() - start) * 1000,
    )
    if tracker is not None:
        tracker.record(usage)
    yield {"type": "usage", **usage.as_dict()}
