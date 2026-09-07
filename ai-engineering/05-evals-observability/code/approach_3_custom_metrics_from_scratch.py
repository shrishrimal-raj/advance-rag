import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 3 - RAG metrics from scratch (pure python, no LLM).

Offline proxies for the core RAGAS metrics, so you can see the math and run
evals in CI without an LLM. Includes a self-check.
"""
import re

STOP = set("the a an is are was were to of and or in on for with your you we our this that it as be".split())


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def content_ngrams(text, n=3):
    ws = re.findall(r"[a-z0-9]+", text.lower())
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)] if len(ws) >= n else []


def context_recall(gold_facts, contexts):
    """fraction of gold facts whose key tokens appear in the retrieved contexts."""
    if not gold_facts:
        return 1.0
    ctx = " ".join(contexts).lower()
    hit = 0
    for fact in gold_facts:
        ft = [w for w in tokens(fact) if w not in STOP]
        if ft and all(w in ctx for w in ft):
            hit += 1
    return hit / len(gold_facts)


def answer_correctness(answer, expected):
    """token overlap between answer and expected answer (coverage of expected)."""
    a, e = tokens(answer), tokens(expected)
    if not e:
        return 1.0
    return len(a & e) / len(e)


def faithfulness_proxy(answer, contexts):
    """fraction of answer content-ngrams supported by the context."""
    ctx = " ".join(contexts)
    ng = content_ngrams(answer, 3)
    if not ng:
        aw = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if w not in STOP]
        if not aw:
            return 1.0
        cset = set(re.findall(r"[a-z0-9]+", ctx.lower()))
        return sum(1 for w in aw if w in cset) / len(aw)
    ctx_grams = set(content_ngrams(ctx, 3))
    return sum(1 for g in ng if g in ctx_grams) / len(ng)


CASES = [
    {"id": "good",
     "expected": "You can return within 30 days for a full refund.",
     "gold_facts": ["30 days", "full refund"],
     "contexts": ["Acme refund policy: orders may be returned within 30 days for a full refund."],
     "answer": "You can return within 30 days for a full refund."},
    {"id": "hallucination",
     "expected": "You can return within 30 days for a full refund.",
     "gold_facts": ["30 days", "full refund"],
     "contexts": ["Acme refund policy: orders may be returned within 30 days for a full refund."],
     "answer": "You can return within 90 days for a double refund plus a gift card."},
]


def score_case(c):
    return {
        "context_recall": round(context_recall(c["gold_facts"], c["contexts"]), 3),
        "answer_correctness": round(answer_correctness(c["answer"], c["expected"]), 3),
        "faithfulness": round(faithfulness_proxy(c["answer"], c["contexts"]), 3),
    }


def main():
    print("=== Approach 3: RAG metrics from scratch (offline) ===")
    scores = {}
    for c in CASES:
        s = score_case(c)
        scores[c["id"]] = s
        print(f"  {c['id']:14} recall={s['context_recall']:.2f} correctness={s['answer_correctness']:.2f} faithfulness={s['faithfulness']:.2f}")
    good, hall = scores["good"], scores["hallucination"]
    assert good["faithfulness"] > hall["faithfulness"], "hallucinated answer must have lower faithfulness"
    assert good["context_recall"] >= hall["context_recall"], "same context -> same recall"
    assert good["answer_correctness"] > hall["answer_correctness"], "correct answer must beat wrong one"
    print("self-check OK: metrics separate good from hallucinated answers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
