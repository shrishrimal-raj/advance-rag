import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 3 - RAG retrieval from scratch (pure numpy, no model, no API).

Implements the actual math behind vector retrieval using a lightweight
TF-IDF vectorizer + cosine similarity + top-k. Runs 100% offline with only
numpy - this is the "understand what the vector DB is doing" version.
"""
import math
import re

import numpy as np


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_idf(docs_tokens):
    n = len(docs_tokens)
    df = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def tfidf_vector(tokens, idf):
    vec = np.zeros(len(idf))
    if not tokens:
        return vec
    idx = {t: i for i, t in enumerate(idf)}
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    m = max(tf.values())
    for t, c in tf.items():
        if t in idx:
            vec[idx[t]] = ((c / m) + 1) * idf[t]
    return vec


def cosine_topk(qvec, mat, k):
    qn = qvec / (np.linalg.norm(qvec) + 1e-9)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


DOCS = [
    "Acme refund policy: orders may be returned within 30 days for a full refund.",
    "Acme shipping: standard delivery takes 3 to 5 business days.",
    "Acme warranty: all hardware carries a 12 month limited warranty.",
    "Globex support hours are Monday to Friday, 9am to 6pm.",
]


def main():
    print("=== Approach 3: from-scratch cosine retrieval (offline) ===")
    toks = [tokenize(d) for d in DOCS]
    idf = build_idf(toks)
    mat = np.vstack([tfidf_vector(t, idf) for t in toks])

    question = "what is the refund policy"
    qvec = tfidf_vector(tokenize(question), idf)
    topk = cosine_topk(qvec, mat, k=3)

    print(f"question   = {question!r}")
    for rank, (i, score) in enumerate(topk, 1):
        print(f"  #{rank} score={score:.4f}  {DOCS[i][:70]}")

    assert topk[0][0] == 0, f"expected refund doc first, got {topk[0][0]}"
    print("self-check OK: refund passage ranked #1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
