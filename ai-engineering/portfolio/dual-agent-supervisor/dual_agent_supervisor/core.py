"""The Dual-Agent Supervisor core.

A secure **maker-checker** multi-agent system with **per-agent provider failover**.
The maker produces an output; the checker validates it against a policy; on rejection
the maker retries (bounded). Each agent is wrapped with a primary + fallback provider,
so a live provider outage is survived by failing over — the system keeps working.
"""
from dataclasses import dataclass
from typing import Callable, List


class ProviderOutage(Exception):
    """Raised by a provider when it is unavailable."""


@dataclass
class CheckerVerdict:
    approved: bool
    reason: str = ""


def make_failover(primary: Callable, fallback: Callable,
                  names=("primary", "fallback")) -> Callable:
    """Wrap a callable so it tries `primary` and, on ANY exception, fails over to
    `fallback`. Returns `(result, provider_name)`."""

    def call(*a, **k):
        try:
            return primary(*a, **k), names[0]
        except Exception:  # noqa: BLE001 - deliberate broad catch for failover
            return fallback(*a, **k), names[1]

    return call


class DualAgentSupervisor:
    """Runs the maker-checker loop with bounded retries and full tracing."""

    def __init__(self, maker: Callable, checker: Callable, max_retries: int = 3):
        self.maker = maker      # (task) -> (value, provider)
        self.checker = checker   # (candidate) -> (CheckerVerdict, provider)
        self.max_retries = max_retries
        self.trace: List[dict] = []

    def run(self, task: str) -> dict:
        for attempt in range(1, self.max_retries + 1):
            value, mprov = self.maker(task)
            verdict, cprov = self.checker(value)
            self.trace.append({"attempt": attempt, "maker": value, "provider": mprov,
                               "approved": verdict.approved, "reason": verdict.reason})
            if verdict.approved:
                return {"ok": True, "output": value, "attempts": attempt,
                        "maker_provider": mprov, "checker_provider": cprov}
        last = self.trace[-1]
        return {"ok": False, "output": "", "attempts": self.max_retries,
                "reason": last.get("reason") or "max_retries_exhausted"}
