"""Lightweight production monitoring for RAG (Module 11) — standalone, no LLM required.

PipelineMetrics records, per request:
    - per-stage wall-clock latency (embed / retrieve / rerank / generate / ...)
    - token counts
    - cache-hit status
    - error status

It exposes two ways to instrument code:
    - a context manager:   with metrics.stage("retrieve"): ...
    - a decorator:         @metrics.timed("generate")

and aggregates everything into a stats report (mean/p95 latency per stage, overall hit rate,
error rate, token totals) printed as rich tables. This mirrors what you'd ship to a dashboard
in production (LangSmith / OpenTelemetry), but with zero external dependencies.

Run:  uv run python 11-production-rag/code/monitoring.py
"""
from __future__ import annotations

import sys
import time
import uuid
import random
import pathlib
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))  # -> project root

from rich.console import Console
from rich.table import Table

# Force UTF-8 output so emoji/box-drawing render correctly on Windows consoles (cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

console = Console()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class RequestRecord:
    request_id: str
    stages: dict[str, float] = field(default_factory=dict)   # stage -> seconds
    tokens: int = 0
    cache_hit: bool = False
    error: bool = False

    @property
    def total_latency(self) -> float:
        return sum(self.stages.values())


def _pct(values: list[float], pct: float) -> float:
    """Percentile (nearest-rank). pct in [0,1]."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((pct / 100.0) * len(s))) - 1))
    return s[idx]


# --------------------------------------------------------------------------- #
# Collector
# --------------------------------------------------------------------------- #
class PipelineMetrics:
    """Collects per-request, per-stage metrics and produces aggregate reports."""

    def __init__(self) -> None:
        self.records: list[RequestRecord] = []
        self._current: RequestRecord | None = None

    # -- request lifecycle -------------------------------------------------- #
    def begin_request(self, request_id: str | None = None) -> RequestRecord:
        self._current = RequestRecord(request_id=request_id or uuid.uuid4().hex[:8])
        return self._current

    def end_request(self, tokens: int = 0, cache_hit: bool = False, error: bool = False) -> None:
        if self._current is not None:
            self._current.tokens = tokens
            self._current.cache_hit = cache_hit
            self._current.error = error
            self.records.append(self._current)
            self._current = None

    # -- instrumentation ---------------------------------------------------- #
    @contextmanager
    def stage(self, name: str):
        """Time a block of code as `name` on the current request."""
        if self._current is None:
            self.begin_request()
        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            if self._current is not None:
                self._current.error = True
            raise
        finally:
            dt = time.perf_counter() - t0
            if self._current is not None:
                self._current.stages[name] = self._current.stages.get(name, 0.0) + dt

    def timed(self, name: str):
        """Decorator form of stage(): @metrics.timed('generate')"""
        def deco(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                with self.stage(name):
                    return fn(*args, **kwargs)
            return wrapper
        return deco

    # -- aggregation -------------------------------------------------------- #
    def report(self) -> dict:
        n = len(self.records)
        totals = [r.total_latency for r in self.records]
        stage_names = sorted({s for r in self.records for s in r.stages})
        per_stage = {}
        for s in stage_names:
            vals = [r.stages[s] for r in self.records if s in r.stages]
            per_stage[s] = {
                "count": len(vals),
                "mean": sum(vals) / len(vals) if vals else 0.0,
                "p95": _pct(vals, 95),
            }
        return {
            "requests": n,
            "mean_total": sum(totals) / n if n else 0.0,
            "p95_total": _pct(totals, 95),
            "hit_rate": sum(r.cache_hit for r in self.records) / n if n else 0.0,
            "error_rate": sum(r.error for r in self.records) / n if n else 0.0,
            "total_tokens": sum(r.tokens for r in self.records),
            "per_stage": per_stage,
        }

    def print_report(self) -> None:
        rep = self.report()
        console.rule(f"📈 Monitoring Report — {rep['requests']} requests")

        t1 = Table(title="Overall")
        t1.add_column("Metric")
        t1.add_column("Value", justify="right")
        t1.add_row("Requests", str(rep["requests"]))
        t1.add_row("Mean total latency", f"{rep['mean_total']:.3f}s")
        t1.add_row("p95 total latency", f"{rep['p95_total']:.3f}s")
        t1.add_row("Cache hit rate", f"{rep['hit_rate']:.0%}")
        t1.add_row("Error rate", f"{rep['error_rate']:.0%}")
        t1.add_row("Total tokens", f"{rep['total_tokens']:,}")
        console.print(t1)

        t2 = Table(title="Per-stage latency")
        t2.add_column("Stage")
        t2.add_column("Count", justify="right")
        t2.add_column("Mean (s)", justify="right")
        t2.add_column("p95 (s)", justify="right")
        for s, m in rep["per_stage"].items():
            t2.add_row(s, str(m["count"]), f"{m['mean']:.3f}", f"{m['p95']:.3f}")
        console.print(t2)


# --------------------------------------------------------------------------- #
# Demo: simulate 20 requests with realistic random latencies
# --------------------------------------------------------------------------- #
def simulate(metrics: PipelineMetrics, n: int = 20) -> None:
    for _ in range(n):
        metrics.begin_request()
        try:
            with metrics.stage("embed"):
                time.sleep(random.uniform(0.03, 0.07))       # ~50ms
            with metrics.stage("retrieve"):
                time.sleep(random.uniform(0.01, 0.03))       # ~20ms
            with metrics.stage("rerank"):
                time.sleep(random.uniform(0.07, 0.13))       # ~100ms
            with metrics.stage("generate"):
                time.sleep(random.uniform(0.8, 2.5))         # 1-3s (dominates)
            tokens = random.randint(200, 900)
            cache_hit = random.random() < 0.30               # ~30% hit rate
            error = False
        except Exception:  # noqa: BLE001
            tokens, cache_hit, error = 0, False, True
        # Occasionally simulate an error path.
        if random.random() < 0.05:
            error = True
        metrics.end_request(tokens=tokens, cache_hit=cache_hit, error=error)


def main() -> None:
    console.rule("📊 Production Monitoring Demo")
    metrics = PipelineMetrics()
    console.print("[dim]Simulating 20 requests with realistic per-stage latencies…[/dim]\n")
    simulate(metrics, n=20)
    metrics.print_report()
    console.print("\n[green]✔ LLM 'generate' dominates latency — caching & model routing pay off.[/green]")


if __name__ == "__main__":
    main()
