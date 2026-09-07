import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 2 - framework-free hybrid search (BM25 + dense), backend switchable.

No LangChain. Dense backend: 'st' (sentence-transformers MiniLM, cached/offline,
default) or 'fastembed' (opt-in; downloads its own model on first use). Fuses
with RRF. Runs offline with the default backend.
"""
import argparse
import re

ITEMS = [
    ("a1", "Acme refund policy: orders may be returned within 30 days for a full refund."),
    ("a2", "Error ERR-4042 means the payment gateway timed out. Retry the charge."),
    ("a3", "How to reset your password: Settings, Reset Password, confirm via email."),
    ("a4", "Acme Express shipping arrives in 1 to 2 business days for an extra fee."),
]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_order(query):
    from rank_bm25 import BM25Okapi
    corpus = [tokenize(t) for _, t in ITEMS]
    if not any(corpus):
        return []
    bm = BM25Okapi(corpus)
    sc = bm.get_scores(tokenize(query))
    return [ITEMS[i][0] for i in sorted(range(len(ITEMS)), key=lambda i: sc[i], reverse=True) if sc[i] > 0]


def dense_order(backend, query):
    import numpy as np
    texts = [t for _, t in ITEMS]
    if backend == "fastembed":
        from fastembed import TextEmbedding
        model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        vecs = np.array(list(model.embed(texts)))
        qv = np.array(list(model.embed([query]))[0])
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = m.encode(texts, normalize_embeddings=True)
        qv = m.encode([query], normalize_embeddings=True)[0]
    qn = qv / (np.linalg.norm(qv) + 1e-9)
    mn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    return [ITEMS[i][0] for i in np.argsort(-sims)]


def rrf(rank_lists, k=60):
    scores = {}
    for ranks in rank_lists:
        for rank, item in enumerate(ranks, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="how do I reset my password")
    ap.add_argument("--backend", choices=["st", "fastembed"], default="st")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    print(f"=== Approach 2: framework-free hybrid (backend={args.backend}) ===")
    print(f"query = {args.query!r}")
    try:
        fused = rrf([bm25_order(args.query), dense_order(args.backend, args.query)])[: args.k]
    except Exception as e:
        print(f"[hybrid] unavailable ({type(e).__name__}: {e})")
        print("Hint: use --backend st for offline; fastembed downloads a model on first use.")
        return 0
    by_id = dict(ITEMS)
    for i, (doc_id, score) in enumerate(fused, 1):
        print(f"  #{i} [{doc_id}] rrf={score:.5f}\n     {by_id[doc_id]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
