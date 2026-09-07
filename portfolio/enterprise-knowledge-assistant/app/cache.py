"""Semantic cache: stores (question embedding, answer) pairs; a lookup hits when
the cosine similarity between the new question and a stored question exceeds
the configured threshold. Avoids paying LLM cost for near-duplicate questions.

The embedder is injected so tests can run fully offline with a fake embedder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class _Entry:
    vector: list[float]
    question: str
    answer: str


@dataclass
class SemanticCache:
    embed_fn: callable  # texts: list[str] -> list[list[float]]
    threshold: float = 0.92
    max_entries: int = 512
    _entries: list[_Entry] = field(default_factory=list, repr=False)
    hits: int = 0
    misses: int = 0

    def get(self, question: str) -> str | None:
        """Return the cached answer if a near-duplicate question exists, else None."""
        vec = self.embed_fn([question])[0]
        best: _Entry | None = None
        best_sim = 0.0
        for entry in self._entries:
            sim = cosine_similarity(vec, entry.vector)
            if sim > best_sim:
                best_sim, best = sim, entry
        if best is not None and best_sim >= self.threshold:
            self.hits += 1
            return best.answer
        self.misses += 1
        return None

    def put(self, question: str, answer: str) -> None:
        vec = self.embed_fn([question])[0]
        # Replace an existing near-duplicate instead of growing unbounded.
        for i, entry in enumerate(self._entries):
            if cosine_similarity(vec, entry.vector) >= self.threshold:
                self._entries[i] = _Entry(vec, question, answer)
                return
        self._entries.append(_Entry(vec, question, answer))
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
