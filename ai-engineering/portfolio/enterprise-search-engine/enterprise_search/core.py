"""The Enterprise Search Engine core.

A **multi-tenant** hybrid search engine: BM25 (sparse) + dense cosine, fused by
Reciprocal Rank Fusion, scoped per tenant (namespace), with metadata filtering and
citation tracking. Pure-python + optional rank_bm25; fully offline-testable.
In production each tenant maps to a Pinecone namespace; here we emulate namespaces
with an in-memory dict keyed by tenant_id.
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
class Doc:
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


class EnterpriseSearchEngine:
    """Multi-tenant hybrid search. Each tenant is an isolated namespace."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self._namespaces: Dict[str, List[Doc]] = {}  # tenant_id -> docs

    def upsert(self, tenant_id: str, docs: List[Doc]):
        self._namespaces.setdefault(tenant_id, []).extend(docs)

    def tenants(self) -> List[str]:
        return list(self._namespaces.keys())

    def _hybrid(self, docs: List[Doc], query: str, qvec: Optional[List[float]],
                top_k: int, filters: Optional[Dict]) -> List[dict]:
        idxs = list(range(len(docs)))
        if filters:
            idxs = [i for i in idxs if all(docs[i].metadata.get(k) == v for k, v in filters.items())]
        if not idxs:
            return []
        sub = [docs[i] for i in idxs]
        qtoks = tokenize(query)
        if _HAVE_BM25:
            corpus = [tokenize(d.text) for d in sub]
            bm = BM25Okapi(corpus) if any(corpus) else None
            sparse = list(bm.get_scores(qtoks)) if bm else [0.0] * len(sub)
        else:  # pragma: no cover
            sparse = [sum(1 for t in qtoks if t in tokenize(d.text)) for d in sub]
        dense = [cosine(qvec, d.vector) for d in sub] if qvec else [0.0] * len(sub)
        fused = {i: 0.0 for i in range(len(sub))}
        for r, i in enumerate(sorted(range(len(sparse)), key=lambda i: sparse[i], reverse=True)):
            fused[i] += 1.0 / (self.rrf_k + r)
        for r, i in enumerate(sorted(range(len(dense)), key=lambda i: dense[i], reverse=True)):
            fused[i] += 1.0 / (self.rrf_k + r)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
        out = []
        for pos, i in enumerate(order):
            d = sub[i]
            out.append({"doc": d, "score": round(fused[i], 6),
                        "citation": {"id": d.id, "source": d.source, "position": pos + 1}})
        return out

    def search(self, tenant_id: str, query: str, query_vector: Optional[List[float]] = None,
               top_k: int = 5, filters: Optional[Dict] = None) -> List[dict]:
        """Hybrid search scoped to one tenant's namespace (hard isolation)."""
        docs = self._namespaces.get(tenant_id, [])
        return self._hybrid(docs, query, query_vector, top_k, filters)
