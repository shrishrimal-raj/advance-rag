"""The Knowledge Engine core.

A production RAG pipeline with **hybrid retrieval** (BM25 sparse + dense cosine,
fused by Reciprocal Rank Fusion), **metadata filtering**, and **citation tracking**.
Pure-python + optional rank_bm25, so it is fully testable offline. In production the
dense vectors come from an embedding model and storage from pgvector; here we use
deterministic stand-ins.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from rank_bm25 import BM25Okapi
    _HAVE_BM25 = True
except Exception:  # pragma: no cover
    _HAVE_BM25 = False


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: Dict = field(default_factory=dict)
    vector: List[float] = field(default_factory=list)


def tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("\n", " ").split() if t]


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class KnowledgeEngine:
    """Hybrid retriever with metadata filtering and citation tracking."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]):
        self.chunks.extend(chunks)

    def _filtered(self, filters: Optional[Dict]) -> List[int]:
        if not filters:
            return list(range(len(self.chunks)))
        return [
            i for i, c in enumerate(self.chunks)
            if all(c.metadata.get(k) == v for k, v in filters.items())
        ]

    @staticmethod
    def _rank(scores: List[float]) -> List[int]:
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def retrieve(self, query: str, query_vector: Optional[List[float]] = None,
                 top_k: int = 5, filters: Optional[Dict] = None) -> List[dict]:
        idxs = self._filtered(filters)
        if not idxs:
            return []
        sub = [self.chunks[i] for i in idxs]
        qtoks = tokenize(query)
        # sparse (BM25, or token-overlap fallback)
        if _HAVE_BM25:
            corpus = [tokenize(c.text) for c in sub]
            bm = BM25Okapi(corpus) if any(corpus) else None
            sparse = list(bm.get_scores(qtoks)) if bm else [0.0] * len(sub)
        else:  # pragma: no cover
            sparse = [sum(1 for t in qtoks if t in tokenize(c.text)) for c in sub]
        # dense (cosine)
        dense = [cosine(query_vector, c.vector) for c in sub] if query_vector else [0.0] * len(sub)
        # Reciprocal Rank Fusion
        fused = {i: 0.0 for i in range(len(sub))}
        for r, i in enumerate(self._rank(sparse)):
            fused[i] += 1.0 / (self.rrf_k + r)
        for r, i in enumerate(self._rank(dense)):
            fused[i] += 1.0 / (self.rrf_k + r)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
        results = []
        for pos, i in enumerate(order):
            c = sub[i]
            results.append({
                "chunk": c,
                "score": round(fused[i], 6),
                "citation": {"id": c.id, "source": c.source, "position": pos + 1},
            })
        return results

    def answer_context(self, query: str, query_vector: Optional[List[float]] = None,
                       top_k: int = 5, filters: Optional[Dict] = None) -> dict:
        """Build the prompt context + ordered citation list for an LLM answer."""
        hits = self.retrieve(query, query_vector, top_k, filters)
        context = "\n\n".join(
            f"[{h['citation']['position']}] ({h['chunk'].source}) {h['chunk'].text}" for h in hits
        )
        return {"context": context, "citations": [h["citation"] for h in hits]}
