"""Hybrid retrieval: BM25 (sparse) + dense vectors, fused with Reciprocal Rank Fusion.

Optional cross-encoder rerank stage (opt-in via RERANK_ENABLED=1; downloads a
~80MB model on first use, so it stays off by default on low-power machines).

Design notes:
- Chroma stores chunks + embeddings persistently; BM25 is rebuilt in memory
  from the same chunk list (cheap for corpus sizes up to ~100k chunks).
- RRF score per doc = sum over lists of 1/(k + rank). We normalize by the
  maximum possible value (two lists, both rank 1) so scores land in (0, 1]
  and a caller-supplied threshold is meaningful.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class RetrievedDoc:
    text: str
    source: str
    score: float  # normalized to (0, 1]

    def as_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "score": round(self.score, 4)}


# ---------------------------------------------------------------------------
# Embeddings (lazy singleton; local MiniLM, cached on disk)
# ---------------------------------------------------------------------------
_embedder = None


def get_embedder():
    """Lazy-load the sentence-transformers embedder. Returns embed(texts)->list[list[float]]."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embedding_model)
        _embedder = model
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_markdown(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split markdown into paragraph-packed chunks of ~size chars."""
    size = size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # Oversized paragraphs are hard-split.
        while len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:size])
            para = para[size - overlap:]
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current.strip():
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    def __init__(self, persist_dir: Path | None = None, collection: str | None = None):
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection or settings.collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._col = self._client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._docs: list[str] = []
        self._sources: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None
        self._reranker = None
        self._load_index()

    # -- index management ---------------------------------------------------
    def _load_index(self) -> None:
        n = self._col.count()
        if n == 0:
            return
        res = self._col.get(limit=n)
        order = sorted(range(len(res["ids"])), key=lambda i: res["ids"][i])
        self._docs = [res["documents"][i] for i in order]
        self._sources = [res["metadatas"][i].get("source", "unknown") for i in order]
        self._id_to_idx = {res["ids"][i]: pos for pos, i in enumerate(order)}
        self._build_bm25()

    def _build_bm25(self) -> None:
        tokens = [_TOKEN_RE.findall(d.lower()) for d in self._docs]
        self._bm25 = BM25Okapi(tokens) if any(tokens) else None

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def ingest_files(self, path: str | Path, reset: bool = False) -> int:
        """Ingest all .md files under `path` into the persistent store. Returns #chunks added."""
        path = Path(path)
        files = sorted(path.rglob("*.md")) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f"No markdown files found under {path}")
        if reset:
            self._col.delete()
            self._docs, self._sources, self._id_to_idx = [], [], {}
            self._bm25 = None
        new_docs: list[str] = []
        new_sources: list[str] = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            for chunk in chunk_markdown(text):
                new_docs.append(chunk)
                new_sources.append(str(f.relative_to(path.parent) if path.is_dir() else f.name))
        if not new_docs:
            return 0
        embeddings = embed_texts(new_docs)
        ids = [f"doc-{len(self._docs) + i}" for i in range(len(new_docs))]
        self._col.upsert(
            ids=ids,
            documents=new_docs,
            embeddings=embeddings,
            metadatas=[{"source": s} for s in new_sources],
        )
        base = len(self._docs)
        self._docs.extend(new_docs)
        self._sources.extend(new_sources)
        for j, doc_id in enumerate(ids):
            self._id_to_idx[doc_id] = base + j
        self._build_bm25()
        return len(new_docs)

    # -- search -------------------------------------------------------------
    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        qv = embed_texts([query])
        res = self._col.query(query_embeddings=qv, n_results=min(k, self.doc_count))
        out: list[tuple[int, float]] = []
        for doc_id, dist in zip(res["ids"][0], res["distances"][0]):
            idx = self._id_to_idx.get(doc_id)
            if idx is not None:
                out.append((idx, 1.0 - dist))  # cosine similarity
        return out

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_TOKEN_RE.findall(query.lower()))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in top if scores[i] > 0]

    def _rrf_fuse(self, *ranked_lists: list[tuple[int, float]]) -> dict[int, float]:
        """Reciprocal Rank Fusion across ranked (index, raw_score) lists."""
        fused: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _raw) in enumerate(ranked, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank)
        max_possible = len(ranked_lists) / (settings.rrf_k + 1)
        return {idx: s / max_possible for idx, s in fused.items()}

    def _rerank(self, query: str, candidates: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not settings.rerank_enabled or not candidates:
            return candidates
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.rerank_model)
        pairs = [[query, c.text] for c in candidates]
        scores = self._reranker.predict(pairs)
        lo, hi = float(min(scores)), float(max(scores))
        span = hi - lo or 1.0
        scored = [(c, (float(s) - lo) / span) for c, s in zip(candidates, scores)]
        scored.sort(key=lambda x: -x[1])
        return [RetrievedDoc(c.text, c.source, s) for c, s in scored]

    def search(self, query: str, k: int = 4) -> list[RetrievedDoc]:
        """Hybrid BM25+dense retrieval with RRF fusion; optional rerank stage."""
        if self.doc_count == 0:
            return []
        k = min(k, self.doc_count)
        dense = self._dense_search(query, k)
        sparse = self._sparse_search(query, k)
        fused = self._rrf_fuse(dense, sparse)
        docs = [
            RetrievedDoc(self._docs[i], self._sources[i], s)
            for i, s in sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        ]
        return self._rerank(query, docs)


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
