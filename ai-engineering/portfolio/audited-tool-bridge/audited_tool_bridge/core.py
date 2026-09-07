"""The Audited Tool Bridge core.

A secure tool gateway exposing internal tools to AI agents with **authentication**,
per-client **rate limiting**, and **full audit logging**. Offline-testable (in-memory
auth + audit); swap in real identity / audit backends in production.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


class TokenBucket:
    """Per-client rate limiter: capacity tokens, refill at refill_per_sec."""

    def __init__(self, capacity: int, refill_per_sec: float, clock=time.monotonic):
        self.capacity = capacity
        self.refill = refill_per_sec
        self._tokens = float(capacity)
        self._last = clock()
        self._clock = clock

    def allow(self) -> bool:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill)
        self._last = now
        if self._tokens >= 1:
            self._tokens -= 1
            return True
        return False


@dataclass
class AuditEntry:
    ts: float
    client: str
    tool: str
    allowed: bool
    reason: str
    args: dict = field(default_factory=dict)

    def as_dict(self):
        return {"ts": round(self.ts, 3), "client": self.client, "tool": self.tool,
                "allowed": self.allowed, "reason": self.reason, "args": self.args}


class ToolBridge:
    """Secure tool gateway: auth -> rate limit -> dispatch, all audited."""

    def __init__(self, valid_keys: Dict[str, str], rate_capacity: int = 5,
                 rate_refill_per_sec: float = 1.0):
        self.valid_keys = valid_keys  # api_key -> client name
        self.tools: Dict[str, Callable] = {}
        self.audit: List[AuditEntry] = []
        self._buckets: Dict[str, TokenBucket] = {}
        self.rate_capacity = rate_capacity
        self.rate_refill = rate_refill_per_sec

    def register_tool(self, name: str, fn: Callable):
        self.tools[name] = fn

    def _bucket(self, client: str) -> TokenBucket:
        if client not in self._buckets:
            self._buckets[client] = TokenBucket(self.rate_capacity, self.rate_refill)
        return self._buckets[client]

    def _log(self, client: str, tool: str, allowed: bool, reason: str, args: dict):
        self.audit.append(AuditEntry(time.time(), client, tool, allowed, reason, args))

    def invoke(self, api_key: str, tool: str, args: Optional[dict] = None) -> dict:
        args = args or {}
        client = self.valid_keys.get(api_key)
        if client is None:
            self._log("unknown", tool, False, "unauthorized", args)
            return {"ok": False, "error": "unauthorized"}
        if not self._bucket(client).allow():
            self._log(client, tool, False, "rate_limited", args)
            return {"ok": False, "error": "rate_limited"}
        fn = self.tools.get(tool)
        if fn is None:
            self._log(client, tool, False, "unknown_tool", args)
            return {"ok": False, "error": f"unknown_tool: {tool}"}
        try:
            result = fn(**args)
            self._log(client, tool, True, "ok", args)
            return {"ok": True, "result": result}
        except Exception as e:  # noqa: BLE001
            self._log(client, tool, True, f"error: {e}", args)
            return {"ok": False, "error": str(e)}

    def audit_log(self, client: Optional[str] = None) -> List[dict]:
        entries = self.audit if client is None else [a for a in self.audit if a.client == client]
        return [e.as_dict() for e in entries]
