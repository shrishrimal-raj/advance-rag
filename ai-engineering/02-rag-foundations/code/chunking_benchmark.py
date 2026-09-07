import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - how chunking strategy affects retrieval quality (offline, no LLM).

Splits one document several ways, indexes each with a from-scratch TF-IDF
retriever, and measures recall@k: does the 'gold' chunk containing the answer
land in the top-k for a set of questions? A cheap proxy for an LLM judge.
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


def tfidf(tokens, idf):
    vec = np.zeros(len(idf))
    idx = {t: i for i, t in enumerate(idf)}
    if not tokens:
        return vec
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    m = max(tf.values())
    for t, c in tf.items():
        if t in idx:
            vec[idx[t]] = ((c / m) + 1) * idf[t]
    return vec


def retrieve(chunks, question, k=3):
    dt = [tokenize(c) for c in chunks]
    idf = build_idf(dt)
    mat = np.vstack([tfidf(t, idf) for t in dt])
    q = tfidf(tokenize(question), idf)
    qn = q / (np.linalg.norm(q) + 1e-9)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    return list(np.argsort(-sims)[:k])


def fixed(text, size, overlap=0):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return [c for c in out if c.strip()]


def recursive_sentence(text, target):
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) > target and cur:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


DOC = (
    "Refunds. Orders may be returned within 30 days for a full refund. "
    "Refunds go to the original payment method within 5 business days. "
    "Shipping. Standard delivery takes 3 to 5 business days. "
    "Express delivery arrives in 1 to 2 business days for an extra fee. "
    "Warranty. All hardware carries a 12 month limited warranty. "
    "Support. Support hours are Monday to Friday, 9am to 6pm."
)

QUESTIONS = [
    ("what is the refund window", "30 days"),
    ("how fast is express shipping", "1 to 2 business days"),
    ("how long is the warranty", "12 month"),
]


def gold_index(chunks, marker):
    for i, c in enumerate(chunks):
        if marker.lower() in c.lower():
            return i
    return -1


def evaluate(name, chunks):
    hits = 0
    for q, marker in QUESTIONS:
        topk = retrieve(chunks, q, k=3)
        g = gold_index(chunks, marker)
        ok = g in topk
        hits += ok
        print(f"    {q!r:40} gold={g} topk={topk} {'HIT' if ok else 'MISS'}")
    recall = hits / len(QUESTIONS)
    print(f"  [{name}] chunks={len(chunks)} recall@3={recall:.2f}")
    return recall


def main():
    print("=== Chunking benchmark (offline, TF-IDF proxy) ===")
    strategies = [
        ("fixed_60", fixed(DOC, 60)),
        ("fixed_150", fixed(DOC, 150)),
        ("fixed_150_ov20", fixed(DOC, 150, overlap=20)),
        ("recursive_120", recursive_sentence(DOC, 120)),
        ("recursive_200", recursive_sentence(DOC, 200)),
    ]
    results = []
    for name, chunks in strategies:
        results.append((name, evaluate(name, chunks)))
    best = max(results, key=lambda x: x[1])
    print(f"\nbest strategy: {best[0]} (recall@3={best[1]:.2f})")
    assert best[1] >= 2 / 3, f"no strategy reached recall@3>=0.66: {results}"
    print("self-check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
