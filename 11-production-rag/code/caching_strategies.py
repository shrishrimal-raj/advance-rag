"""Exact vs semantic cache for RAG (Module 11) — local embeddings only, runs offline.

Compares two cache strategies on a 10-query workload containing exact repeats and
paraphrases:

  ExactCache     — SHA-256 of the normalized query (strip/lower/collapse spaces).
                   O(1), zero false positives, but only catches *identical* repeats.
  SemanticCache  — embed the query, cosine-similarity against stored entries;
                   HIT when sim > threshold (default 0.95). Catches paraphrases,
                   but a threshold that's too low serves WRONG answers (false positives).

For each strategy we report: hit rate, average latency saved (a cache hit skips the
~2 s LLM generation stage), and a false-positive demo: a near-miss query about a
DIFFERENT topic that must NOT be served from cache.

Run:  uv run python 11-production-rag/code/caching_strategies.py
"""
from __future__ import annotations

import sys
import time
import hashlib
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))  # -> project root

from rich.console import Console
from rich.table import Table

console = Console()

# Simulated per-stage latencies (seconds) — a MISS pays all of them, a HIT only embeds.
LATENCY = {"embed": 0.05, "retrieve": 0.02, "rerank": 0.10, "generate": 2.0}
FULL_PIPELINE_MS = sum(LATENCY.values()) * 1000
HIT_COST_MS = LATENCY["embed"] * 1000  # semantic hit still needs one embedding


def normalize(q: str) -> str:
    return " ".join(q.strip().lower().split())


# --------------------------------------------------------------------------- #
# Exact cache
# --------------------------------------------------------------------------- #
class ExactCache:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.hits = self.misses = 0

    def get(self, query: str) -> str | None:
        key = hashlib.sha256(normalize(query).encode("utf-8")).hexdigest()
        ans = self._store.get(key)
        self.hits += ans is not None
        self.misses += ans is None
        return ans

    def put(self, query: str, answer: str) -> None:
        self._store[hashlib.sha256(normalize(query).encode("utf-8")).hexdigest()] = answer

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


# --------------------------------------------------------------------------- #
# Semantic cache (in-memory, cosine over local MiniLM embeddings)
# --------------------------------------------------------------------------- #
class SemanticCache:
    def __init__(self, threshold: float = 0.95):
        from shared.config import get_embeddings
        self.threshold = threshold
        self._emb = get_embeddings()
        self._keys: list[str] = []
        self._vecs: list[list[float]] = []
        self._answers: dict[str, str] = {}
        self.hits = self.misses = 0

    def _cosine(self, a: list[float], b: list[float]) -> float:
        import numpy as np
        va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
        return float(np.dot(va, vb) / denom)

    def get(self, query: str) -> tuple[str | None, float]:
        """Return (answer_or_None, best_similarity)."""
        if not self._keys:
            return None, 0.0
        v = self._emb.embed_query(query)
        sims = [self._cosine(v, u) for u in self._vecs]
        i = max(range(len(sims)), key=sims.__getitem__)
        if sims[i] > self.threshold:
            self.hits += 1
            return self._answers[self._keys[i]], sims[i]
        self.misses += 1
        return None, sims[i]

    def put(self, query: str, answer: str) -> None:
        v = self._emb.embed_query(query)
        self._keys.append(normalize(query))
        self._vecs.append(v)
        self._answers[normalize(query)] = answer

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


# --------------------------------------------------------------------------- #
# Workload: 10 queries — 3 exact repeats, 3 paraphrases, 4 unique
# --------------------------------------------------------------------------- #
WORKLOAD = [
    "What is the capital of France?",            # 1  unique
    "How long is Acme Robotics' warranty?",      # 2  unique
    "What is the capital of France?",            # 3  EXACT repeat of 1
    "What city is France's capital?",            # 4  paraphrase of 1
    "What is the capital of France?",            # 5  EXACT repeat of 1
    "How long does Acme Robotics warranty last?",# 6  paraphrase of 2
    "What is the capital of Germany?",           # 7  unique (near-miss topic!)
    "Tell me about Acme Robotics' warranty.",    # 8  paraphrase of 2
    "What is the capital of Japan?",             # 9  unique
    "What city is France's capital?",            # 10 EXACT repeat of 4
]

ANSWERS = {
    "france": "The capital of France is Paris.",
    "warranty": "Acme Robotics offers a 3-year warranty with 4-hour response time.",
    "germany": "The capital of Germany is Berlin.",
    "japan": "The capital of Japan is Tokyo.",
}


def fake_pipeline(query: str) -> str:
    """Deterministic stand-in for the real RAG pipeline (no LLM needed)."""
    q = query.lower()
    if "france" in q:
        return ANSWERS["france"]
    if "warranty" in q or "acme" in q:
        return ANSWERS["warranty"]
    if "germany" in q:
        return ANSWERS["germany"]
    if "japan" in q:
        return ANSWERS["japan"]
    return f"(generated answer for: {query})"


def run_workload(cache, label: str) -> Table:
    table = Table(title=f"{label} — 10-query workload")
    table.add_column("#", justify="right")
    table.add_column("Query", max_width=44)
    table.add_column("Result")
    table.add_column("Sim", justify="right")
    for i, q in enumerate(WORKLOAD, start=1):
        t0 = time.perf_counter()
        res = cache.get(q)
        lookup_ms = (time.perf_counter() - t0) * 1000
        if isinstance(res, tuple):
            ans, sim = res
        else:
            ans, sim = res, 1.0
        if ans is not None:
            tag = "[green]HIT[/green]"
        else:
            ans = fake_pipeline(q)
            cache.put(q, ans)
            tag = "[yellow]MISS[/yellow]"
        table.add_row(str(i), q, tag, f"{sim:.4f}")
    console.print(table)

    hits = cache.hits
    saved_ms = hits * (FULL_PIPELINE_MS - HIT_COST_MS)
    console.print(f"[bold]{label}[/bold]: hit rate = {cache.hit_rate:.0%} ({hits}/10), "
                  f"avg latency saved per hit ≈ {FULL_PIPELINE_MS - HIT_COST_MS:.0f} ms, "
                  f"total saved ≈ {saved_ms / 1000:.2f} s over the workload\n")
    return table


def main() -> None:
    console.rule("🗄️  Caching Strategies Demo (exact vs semantic)")
    console.print(f"[dim]Simulated full-pipeline cost: {FULL_PIPELINE_MS:.0f} ms; "
                  f"a semantic hit costs ~{HIT_COST_MS:.0f} ms (one embedding).[/dim]\n")

    # -- Embeddings availability (graceful degradation) ---------------------- #
    try:
        sem = SemanticCache(threshold=0.95)
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]Local embeddings unavailable ({e.__class__.__name__}). "
                      f"Showing exact-cache results only.[/yellow]")
        sem = None

    run_workload(ExactCache(), "Exact cache (SHA-256)")
    if sem is not None:
        run_workload(sem, "Semantic cache (cosine > 0.95)")

    # -- False-positive risk demo -------------------------------------------- #
    console.rule("⚠️  False-positive risk demo")
    if sem is not None:
        # Near-miss: same shape as the cached France/Germany/Japan queries, but
        # Italy was NEVER asked — so no correct answer exists in the cache.
        probe = "What is the capital of Italy?"
        ans, sim = sem.get(probe)
        console.print(f"Probe: [italic]{probe}[/italic]  (never asked before)")
        console.print(f"Best similarity to any cached entry: {sim:.4f} (threshold 0.95)")
        if ans is not None:
            console.print(f"[red]FALSE POSITIVE — would serve: {ans!r}[/red]")
        else:
            console.print("[green]Correctly NOT served from cache → falls through to the pipeline.[/green]")
        # Show the danger zone: the SAME probe under a too-low threshold.
        low = 0.60
        verdict = "[red]FALSE POSITIVE — serves a wrong 'capital' answer with no error raised[/red]" \
            if sim > low else "[green]still safe even at 0.60[/green]"
        console.print(f"Same probe if someone lowered the threshold to {low:.2f}: {verdict}")
        console.print("\n[bold]Lesson:[/bold] a lower threshold turns near-misses into silent wrong answers. "
                      "Tune on real traffic; log every hit for audit.")
    else:
        console.print("[dim]Skipped (no embeddings). An exact cache has ZERO false positives by construction.[/dim]")

    console.print("\n[green]✔ Exact = safe & cheap, low hit rate. Semantic = higher hit rate, "
                  "threshold is your safety dial.[/green]")


if __name__ == "__main__":
    main()
