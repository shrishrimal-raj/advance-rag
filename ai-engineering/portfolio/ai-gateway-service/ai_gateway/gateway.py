"""AI Gateway core: provider routing with retry, fallback, rate limiting, structured logging.

A provider is any callable `prompt -> str`. The gateway tries them in order,
retrying each up to `max_retries`, falling back to the next on failure, enforcing a
token-bucket rate limit, and emitting structured log events. If every provider fails
it degrades gracefully instead of raising.
"""
import time
from dataclasses import dataclass
from typing import Callable, List, Optional


class ProviderError(Exception):
    pass


@dataclass
class GatewayConfig:
    max_retries: int = 2
    timeout_s: float = 30.0
    rate_limit_per_min: int = 60


def validate_config(cfg: GatewayConfig):
    problems = []
    if cfg.max_retries < 0:
        problems.append("max_retries must be >= 0")
    if cfg.timeout_s <= 0:
        problems.append("timeout_s must be > 0")
    if cfg.rate_limit_per_min <= 0:
        problems.append("rate_limit_per_min must be > 0")
    if problems:
        raise ValueError("; ".join(problems))


class TokenBucket:
    """Minimal token-bucket rate limiter."""

    def __init__(self, rate_per_min: int):
        self.capacity = rate_per_min
        self.tokens = float(rate_per_min)
        self.refill_per_s = rate_per_min / 60.0
        self.last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_s)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


@dataclass
class GatewayResult:
    ok: bool
    text: str
    provider: Optional[str] = None
    attempts: int = 0
    degraded: bool = False


class AIGateway:
    def __init__(self, providers: List[Callable[[str], str]], config: Optional[GatewayConfig] = None, logger=None):
        if not providers:
            raise ValueError("at least one provider required")
        self.providers = providers
        self.config = config or GatewayConfig()
        validate_config(self.config)
        self.bucket = TokenBucket(self.config.rate_limit_per_min)
        self._log = logger or (lambda level, event, **f: None)

    def complete(self, prompt: str) -> GatewayResult:
        if not self.bucket.allow():
            self._log("warn", "rate_limited", prompt_len=len(prompt))
            return GatewayResult(ok=False, text="rate limited", degraded=True)
        last_err = None
        for i, prov in enumerate(self.providers):
            name = getattr(prov, "__name__", f"provider_{i}")
            for attempt in range(self.config.max_retries + 1):
                try:
                    text = prov(prompt)
                    self._log("info", "completed", provider=name, attempt=attempt + 1)
                    return GatewayResult(ok=True, text=text, provider=name, attempts=attempt + 1)
                except Exception as e:  # noqa: BLE001 - gateway must survive any provider error
                    last_err = e
                    self._log("warn", "provider_error", provider=name, attempt=attempt + 1, error=str(e))
        self._log("error", "all_providers_failed", error=str(last_err))
        return GatewayResult(ok=False, text=f"degraded: {last_err}", degraded=True)
