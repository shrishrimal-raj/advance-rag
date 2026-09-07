"""Semantic cache for RAG (Module 11) — standalone, no LLM required.

A SemanticCache wraps a *generator function* (query -> answer). On each call it:
    1. embeds the query,
    2. searches a small Chroma "cache" collection for the nearest stored entry,
    3. if cosine similarity > threshold (default 0.95) -> return the cached answer (HIT),
    4. else run the wrapped function, store (query, answer), and return it (MISS).

Why semantic (not exact-match)? Real traffic is full of paraphrases. Embedding the query and
matching near-duplicates catches "What's the capital of France?" vs "What city is France's
capital?" — the common case an exact hash would miss.

Chroma note: cosine *distance* = 1 - similarity, so a HIT means distance < (1 - threshold).

Run:  uv run python 11-production-rag/code/semantic_cache.py
"""
from __future__ import annotations

import sys
import pathlib
import hashlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))  # -> project root

from rich.console import Console
from rich.table import Table

from shared.config import get_embeddings

# Force UTF-8 output so emoji/box-drawing render correctly on Windows consoles (cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

console = Console()

# Use a dedicated persist dir so the cache never collides with the Papeer index.
CACHE_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "chroma_db"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_COLLECTION = "rag_semantic_cache"


class SemanticCache:
    """Chroma-backed semantic cache that wraps a generator function."""

    def __init__(self, threshold: float = 0.95, collection: str = CACHE_COLLECTION):
        from chromadb import PersistentClient
        self.threshold = threshold
        self.embeddings = get_embeddings()
        self.client = PersistentClient(str(CACHE_DIR))
        # Reuse-or-create; cosine space so distance = 1 - similarity.
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()[:16]

    def lookup(self, query: str) -> tuple[str | None, float]:
        """Return (cached_answer_or_None, similarity_of_nearest_entry)."""
        if self.col.count() == 0:
            return None, 0.0
        res = self.col.query(query_embeddings=[self._embed(query)], n_results=1)
        if not res["ids"] or not res["ids"][0]:
            return None, 0.0
        distance = float(res["distances"][0][0])
        similarity = 1.0 - distance          # cosine space
        if similarity > self.threshold:
            answer = res["metadatas"][0][0].get("answer")
            return answer, similarity
        return None, similarity

    def store(self, query: str, answer: str) -> None:
        self.col.upsert(
            ids=[self._key(query)],
            documents=[query],
            embeddings=[self._embed(query)],
            metadatas=[{"answer": answer}],
        )

    def wrap(self, fn):
        """Wrap a generator function with a semantic fast-path."""
        def cached(query: str) -> dict:
            hit, sim = self.lookup(query)
            if hit is not None:
                self.hits += 1
                return {"answer": hit, "cached": True, "similarity": round(sim, 4)}
            answer = fn(query)               # MISS -> compute
            self.store(query, answer)
            self.misses += 1
            return {"answer": answer, "cached": False, "similarity": round(sim, 4)}
        return cached

    def clear(self) -> None:
        """Invalidation: drop all cached entries (e.g., after a KB update)."""
        try:
            self.client.delete_collection(CACHE_COLLECTION)
        except Exception:  # noqa: BLE001
            pass
        self.col = self.client.get_or_create_collection(
            name=CACHE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# --------------------------------------------------------------------------- #
# Demo: a fake "generator" (no LLM) + 3 queries where #3 paraphrases #1
# --------------------------------------------------------------------------- #
def fake_answer_generator(query: str) -> str:
    """Stands in for the real RAG pipeline. Deterministic so the demo is reproducible."""
    q = query.lower()
    if "capital" in q and "france" in q:
        return "The capital of France is Paris."
    if "warranty" in q or "acme" in q:
        return "Acme Robotics offers a 3-year warranty with 4-hour response time."
    return f"(generated answer for: {query})"


def main() -> None:
    console.rule("🧠 Semantic Cache Demo")
    cache = SemanticCache(threshold=0.95)
    cache.clear()                       # start from a clean cache
    ask = cache.wrap(fake_answer_generator)

    queries = [
        "What is the capital of France?",                 # #1 -> MISS (cold cache)
        "How long is Acme Robotics' warranty?",           # #2 -> MISS (different topic)
        "What is the capital city of France?",            # #3 -> paraphrase of #1 -> HIT
    ]

    table = Table(title="Semantic cache results")
    table.add_column("#", justify="right")
    table.add_column("Query", max_width=45)
    table.add_column("Result")
    table.add_column("Similarity", justify="right")
    table.add_column("Answer", max_width=55)

    for i, q in enumerate(queries, start=1):
        r = ask(q)
        tag = "[green]HIT[/green]" if r["cached"] else "[yellow]MISS[/yellow]"
        table.add_row(str(i), q, tag, f"{r['similarity']:.4f}", r["answer"])
        console.print(f"[dim]#{i} {q} -> {tag} (sim={r['similarity']:.4f})[/dim]")

    console.print(table)
    console.print(f"\nCache stats: hits={cache.hits} misses={cache.misses} "
                  f"hit_rate={cache.hit_rate:.0%}")
    console.print("[green]✔ Query #3 (a paraphrase of #1) was served from cache.[/green]")


if __name__ == "__main__":
    main()
