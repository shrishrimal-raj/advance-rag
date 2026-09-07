"""Phase 2 — Hybrid retrieval + cross-encoder reranking.

Pipeline (the industry-standard two-stage retrieve-then-rerank):
    1. DENSE  : Chroma cosine search  -> top-10 (semantic recall)
    2. SPARSE : BM25 over the sidecar -> top-10 (exact-term recall)
    3. FUSE   : Reciprocal Rank Fusion (k=60) over the two ranked lists
    4. RERANK : cross-encoder scores the fused candidates jointly -> top-3

Why hybrid? Dense captures meaning; BM25 captures exact tokens (SKUs, acronyms, error codes).
Why RRF? It merges two differently-scaled rank lists without score normalization.
Why rerank? A cross-encoder scores the (query, passage) *pair* jointly — far more precise than
bi-encoders, but only affordable on a small candidate set. Hence the two-stage design.

Run:  uv run python 10-capstone-project/code/papeer/retrieval.py "your question"
"""
from __future__ import annotations

import json
import sys
import pathlib

_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from rich.console import Console
from rich.table import Table

from papeer.config import (
    BM25_TOP_K,
    CHUNKS_JSON,
    DENSE_TOP_K,
    RERANK_TOP_K,
    RRF_K,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    get_embeddings,
    get_reranker,
)

# Force UTF-8 output so emoji/box-drawing render correctly on Windows consoles (cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

console = Console()

# Process-level caches: loading the cross-encoder / embeddings / Chroma client per call
# would cost 10-15s of model I/O on every query — unacceptable latency for a request path.
_RERANKER = None
_CHROMA_CLIENT = None
_EMBEDDINGS = None


def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = get_embeddings()
    return _EMBEDDINGS


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = get_reranker()
    return _RERANKER


def _get_chroma_client():
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        from chromadb import PersistentClient
        _CHROMA_CLIENT = PersistentClient(str(CHROMA_DIR))
    return _CHROMA_CLIENT


def _load_sidecar() -> dict[str, dict]:
    """id -> {text, source, ...} from the BM25 sidecar written by ingestion."""
    if not CHUNKS_JSON.exists():
        raise FileNotFoundError(
            f"{CHUNKS_JSON} not found — run ingestion first "
            f"(uv run python 10-capstone-project/code/papeer/ingestion.py)"
        )
    data = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    return {d["id"]: d for d in data}


def _dense_search(question: str, n: int, where: dict | None = None) -> list[str]:
    """Return chunk ids from Chroma cosine search, best-first.

    We embed the query with the SAME model used at ingestion time (via query_embeddings)
    rather than letting Chroma use its default embedding function — mixing encoders would
    silently degrade recall because the vectors live in different spaces.

    `where` is an optional Chroma metadata filter (e.g. {"source": "products.csv"}) applied
    as a pre-filter so filtered-out chunks never enter the candidate set.
    """
    client = _get_chroma_client()
    col = client.get_collection(CHROMA_COLLECTION)
    qv = _get_embeddings().embed_query(question)
    kwargs = {"where": where} if where else {}
    res = col.query(query_embeddings=[qv], n_results=n, **kwargs)
    return list(res["ids"][0])


def _bm25_search(question: str, docs_by_id: dict[str, dict], n: int) -> list[str]:
    """Return chunk ids ranked by BM25, best-first."""
    from rank_bm25 import BM25Okapi
    ids = list(docs_by_id.keys())
    corpus_tokens = [docs_by_id[i]["text"].split() for i in ids]
    if not any(corpus_tokens):
        return []
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(question.split())
    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:n]
    return [ids[i] for i in ranked if scores[i] > 0]


def _rrf_fuse(dense_ids: list[str], sparse_ids: list[str]) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank). Returns [(id, score)] desc."""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def retrieve(
    question: str,
    top_k: int | None = None,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Hybrid retrieve + rerank. Returns top passages: {text, source, page, score, rank}.

    `metadata_filter` (optional) restricts retrieval to chunks whose metadata matches ALL
    given key/value pairs (Chroma `where` semantics), e.g. {"source": "products.csv"} or
    {"doc_type": "MD"}. Applied to BOTH the dense and sparse legs so the fusion stays honest.
    """
    top_k = top_k or RERANK_TOP_K
    docs_by_id = _load_sidecar()
    if metadata_filter:
        docs_by_id = {
            cid: d for cid, d in docs_by_id.items()
            if all(d.get(k) == v for k, v in metadata_filter.items())
        }

    dense_ids = _dense_search(question, DENSE_TOP_K, where=metadata_filter)
    sparse_ids = _bm25_search(question, docs_by_id, BM25_TOP_K)
    fused = _rrf_fuse(dense_ids, sparse_ids)

    # Take a generous fused candidate set for the reranker (union of both lists).
    candidates = [cid for cid, _ in fused][: max(DENSE_TOP_K, BM25_TOP_K)]
    if not candidates:
        return []

    # Cross-encoder rerank: score each (question, passage) pair jointly.
    reranker = _get_reranker()
    pairs = [(question, docs_by_id[c]["text"]) for c in candidates]
    try:
        ce_scores = reranker.predict(pairs)
    except Exception as e:  # noqa: BLE001 - if reranker unavailable, fall back to RRF order
        console.print(f"[yellow]⚠️  reranker failed ({e}); using RRF order[/yellow]")
        ce_scores = [float(len(candidates) - i) for i in range(len(candidates))]

    order = sorted(range(len(candidates)), key=lambda i: ce_scores[i], reverse=True)[:top_k]
    results = []
    for new_rank, i in enumerate(order, start=1):
        cid = candidates[i]
        doc = docs_by_id[cid]
        results.append({
            "text": doc["text"],
            "source": doc["source"],
            "page": doc.get("page"),
            "score": float(ce_scores[i]),
            "rank": new_rank,
        })
    return results


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What navigation does the Falcon AMR use?"
    hits = retrieve(q)
    table = Table(title=f"🔎 Hybrid Retrieval — {q!r}")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Passage", max_width=70)
    for h in hits:
        table.add_row(str(h["rank"]), f"{h['score']:.3f}", h["source"], h["text"][:140] + "…")
    console.print(table)
