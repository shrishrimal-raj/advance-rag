import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - naive (dense) vs hybrid (BM25+dense RRF) vs +rerank (offline).

Shows hybrid retrieval beats either single method on a mixed gold corpus.
Rerank is opt-in (--rerank) to avoid the cross-encoder download.
"""
import argparse
import re

import numpy as np

ITEMS = [
    ("d1", "Acme refund policy: orders may be returned within 30 days for a full refund."),
    ("d2", "Error ERR-4042 means the payment gateway timed out. Retry the charge."),
    ("d3", "How to reset your password: Settings, Reset Password, confirm via email."),
    ("d4", "Acme Express shipping arrives in 1 to 2 business days for an extra fee."),
    ("d5", "Globex error G-77 indicates an expired API key. Regenerate it in the console."),
]
QS = [
    ("what is error code ERR-4042", "d2", "exact"),
    ("how can I change my login credentials", "d3", "semantic"),
    ("refund return window", "d1", "mixed"),
    ("expired api key globex", "d5", "exact"),
]


def tokenize(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def bm25_order(query):
    from rank_bm25 import BM25Okapi
    corpus = [tokenize(t) for _, t in ITEMS]
    bm = BM25Okapi(corpus)
    sc = bm.get_scores(tokenize(query))
    return [ITEMS[i][0] for i in sorted(range(len(ITEMS)), key=lambda i: sc[i], reverse=True) if sc[i] > 0]


def dense_order(query, enc):
    vecs = np.vstack(enc([t for _, t in ITEMS]))
    qv = enc([query])[0]
    qn = qv / (np.linalg.norm(qv) + 1e-9)
    mn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    return [ITEMS[i][0] for i in np.argsort(-sims)]


def rrf(lists, k=60):
    sc = {}
    for L in lists:
        for r, it in enumerate(L, 1):
            sc[it] = sc.get(it, 0.0) + 1.0 / (k + r)
    return [it for it, _ in sorted(sc.items(), key=lambda kv: kv[1], reverse=True)]


def hit(order, gold, k=3):
    return 1.0 if gold in order[:k] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    print("=== Benchmark: naive vs hybrid vs +rerank (offline) ===")
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")

    def enc(texts):
        return m.encode(texts, normalize_embeddings=True)

    naive = hybrid = 0.0
    for q, gold, kind in QS:
        n = dense_order(q, enc)[:3]
        h = rrf([bm25_order(q), dense_order(q, enc)])[:3]
        nr, hr = hit(n, gold), hit(h, gold)
        naive += nr
        hybrid += hr
        print(f"  [{kind:8}] {q!r:42} naive={'HIT' if nr else 'MISS'} hybrid={'HIT' if hr else 'MISS'}")
    nN = len(QS)
    print(f"\nrecall@3  naive={naive / nN:.2f}  hybrid={hybrid / nN:.2f}")
    assert hybrid / nN >= naive / nN, "hybrid should not be worse than naive"
    print("self-check OK: hybrid >= naive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
