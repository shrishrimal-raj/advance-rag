"""Evaluation metrics.

Two tiers:

1. CUSTOM OFFLINE METRICS — no API keys, no network, deterministic:
   - context_keyword_coverage : do the retrieved docs contain the question's key terms?
   - answer_relevancy         : embedding similarity between question and answer
   - faithfulness_proxy       : mean per-sentence similarity of the answer to the context

2. OPTIONAL RAGAS METRICS — real LLM-judged faithfulness / answer relevancy.
   Wrapped so that a missing ragas install or missing LLM key skips gracefully
   (returns an empty dict) instead of crashing the run.
"""
from __future__ import annotations

import re
from typing import Callable, Protocol

from .config import cosine

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

STOPWORDS = frozenset(
    """a an the and or but if then else of in on at to for from by with about into
    over under is are was were be been being do does did done can could should would
    will shall may might must what which who whom when where why how this that these
    those it its as not no nor so than too very just also there here""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())]


def key_terms(text: str) -> list[str]:
    """Question tokens minus stopwords (deduped, order preserved)."""
    seen: dict[str, None] = {}
    for tok in tokenize(text):
        if tok not in STOPWORDS:
            seen.setdefault(tok, None)
    return list(seen)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# 1. Custom offline metrics — each returns a score in [0, 1]
# ---------------------------------------------------------------------------

def context_keyword_coverage(question: str, contexts: list[str]) -> float:
    """Fraction of the question's key terms present in the retrieved context."""
    terms = key_terms(question)
    if not terms:
        return 0.0
    haystack = " ".join(contexts).lower()
    hits = sum(1 for t in terms if t in haystack)
    return _clamp01(hits / len(terms))


def answer_relevancy(question: str, answer: str, embed: Embedder) -> float:
    """Embedding similarity between the question and the generated answer."""
    if not answer.strip():
        return 0.0
    return _clamp01(cosine(embed.embed(question), embed.embed(answer)))


def faithfulness_proxy(answer: str, contexts: list[str], embed: Embedder) -> float:
    """Claim-context similarity proxy.

    Splits the answer into sentence-level 'claims' and scores each by its best
    cosine similarity to any retrieved context chunk; returns the mean. A fully
    grounded answer scores high; a hallucinated one drifts toward 0.
    """
    claims = split_sentences(answer)
    if not claims or not contexts:
        return 0.0
    ctx_vecs = [embed.embed(c) for c in contexts]
    scores = []
    for claim in claims:
        cv = embed.embed(claim)
        scores.append(max(cosine(cv, c) for c in ctx_vecs))
    return _clamp01(sum(scores) / len(scores))


def evaluate_question(
    question: str,
    contexts: list[str],
    answer: str,
    embed: Embedder,
) -> dict[str, float]:
    """All custom offline metrics for one question. Values in [0, 1]."""
    return {
        "context_keyword_coverage": round(context_keyword_coverage(question, contexts), 4),
        "answer_relevancy": round(answer_relevancy(question, answer, embed), 4),
        "faithfulness_proxy": round(faithfulness_proxy(answer, contexts, embed), 4),
    }


# ---------------------------------------------------------------------------
# 2. Optional RAGAS metrics (graceful skip)
# ---------------------------------------------------------------------------

def ragas_scores(
    question: str,
    reference_answer: str,
    contexts: list[str],
    answer: str,
    llm,
    embedder=None,
) -> dict[str, float]:
    """LLM-judged RAGAS metrics. Returns {} (skip) when ragas/LLM unavailable.

    Never raises: any import/runtime failure degrades to an empty dict so the
    rest of the run proceeds with the custom offline metrics.
    """
    try:
        from langchain_core.messages import HumanMessage
        from ragas import evaluate
        from ragas.metrics import Faithfulness, AnswerRelevancy
        from datasets import Dataset
    except Exception:
        return {}
    if llm is None:
        return {}
    try:
        single = {"question": question, "answer": answer,
                  "ground_truth": reference_answer, "contexts": contexts}
        ds = Dataset.from_list([single])
        result = evaluate(ds, metrics=[Faithfulness(), AnswerRelevancy()], llm=llm)
        row = result.to_pandas().iloc[0]
        out = {}
        for name in ("faithfulness", "answer_relevancy"):
            val = row.get(name)
            if val is not None and not (isinstance(val, float) and val != val):  # NaN check
                out[f"ragas_{name}"] = round(float(val), 4)
        return out
    except Exception:
        return {}
