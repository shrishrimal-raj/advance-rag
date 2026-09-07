import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The Enterprise Search Engine (Week 3 weekly build).

Multi-tenant hybrid search: BM25 (keyword) + dense (semantic) -> RRF fusion ->
optional cross-encoder re-rank. Returns ranked results with citations. The core
search needs NO LLM (rerank is opt-in via --rerank).

Run:
    python main.py --tenant acme --query "ERR-4042"
    python main.py --tenant acme --query "how do I reset my password" --rerank
"""
import argparse
import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import CHROMA_DIR  # noqa: E402,F401

CORPUS = {
    "acme": [
        {"id": "acme-1", "text": "Acme refund policy: orders may be returned within 30 days for a full refund."},
        {"id": "acme-2", "text": "Error ERR-4042 means the payment gateway timed out. Retry the charge or contact billing."},
        {"id": "acme-3", "text": "How to reset your password: go to Settings, choose Reset Password, and confirm via email."},
        {"id": "acme-4", "text": "Acme Express shipping arrives in 1 to 2 business days for an extra fee."},
        {"id": "acme-5", "text": "Acme hardware warranty covers manufacturing defects for 12 months."},
    ],
    "globex": [
        {"id": "globex-1", "text": "Globex support hours are Monday to Friday, 9am to 6pm local time."},
        {"id": "globex-2", "text": "Globex error G-77 indicates an expired API key. Regenerate it in the console."},
    ],
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def make_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def enc(texts):
        return model.encode(texts, normalize_embeddings=True).tolist()

    return enc


def bm25_rank(passages, query):
    from rank_bm25 import BM25Okapi
    corpus = [tokenize(p["text"]) for p in passages]
    if not any(corpus):
        return []
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    return [passages[i]["id"] for i in order if scores[i] > 0]


def dense_rank(vecs, ids, query_vec):
    import numpy as np
    mat = np.vstack(vecs)
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    return [ids[i] for i in np.argsort(-sims)]


def rrf(rank_lists, k=60):
    scores = {}
    for ranks in rank_lists:
        for rank, item in enumerate(ranks, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rerank(query, by_id, cand_ids):
    from shared.config import get_reranker
    model = get_reranker()
    pairs = [[query, by_id[i]["text"]] for i in cand_ids]
    scores = model.predict(pairs)
    order = sorted(range(len(cand_ids)), key=lambda j: scores[j], reverse=True)
    return [(cand_ids[j], float(scores[j])) for j in order]


def search(query, tenant="acme", k=3, use_rerank=False, fetch=10):
    passages = CORPUS.get(tenant, [])
    if not passages:
        return []
    by_id = {p["id"]: p for p in passages}
    enc = make_embedder()
    vecs = enc([p["text"] for p in passages])
    qv = enc([query])[0]
    ids = [p["id"] for p in passages]

    b_list = bm25_rank(passages, query)
    d_list = dense_rank(vecs, ids, qv)
    fused = rrf([b_list, d_list])[:fetch]

    if use_rerank:
        rr = rerank(query, by_id, [i for i, _ in fused])
        return [(i, s, by_id[i]["text"]) for i, s in rr[:k]]
    return [(i, s, by_id[i]["text"]) for i, s in fused[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="ERR-4042")
    ap.add_argument("--tenant", default="acme")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()

    print(f"=== Enterprise Search Engine (tenant={args.tenant}) ===")
    print(f"query = {args.query!r}  rerank={args.rerank}")
    try:
        results = search(args.query, args.tenant, args.k, args.rerank)
    except Exception as e:
        print(f"[search] unavailable ({type(e).__name__}: {e})")
        print("Hint: local MiniLM must be cached; --rerank needs a one-time model download.")
        return 0
    for i, (doc_id, score, text) in enumerate(results, 1):
        print(f"  #{i} [{doc_id}] score={score:.4f}\n     {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
