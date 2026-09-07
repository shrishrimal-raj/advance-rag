"""Offline tests for the evaluation harness (no LLM, no network)."""
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.config import HashingEmbedder, cosine
from harness.metrics import (
    answer_relevancy,
    context_keyword_coverage,
    evaluate_question,
    faithfulness_proxy,
    key_terms,
    ragas_scores,
    tokenize,
)

EMB = HashingEmbedder()


def test_tokenize_lowercases_and_strips():
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_key_terms_subset_of_tokens():
    terms = key_terms("chunking strategies for chunking documents")
    assert "chunking" in terms
    assert len(terms) <= 4


def test_coverage_perfect_when_all_terms_present():
    assert context_keyword_coverage("hnsw graph search", ["HNSW uses a graph for search."]) == pytest.approx(1.0)


def test_coverage_zero_when_disjoint():
    assert context_keyword_coverage("quantum entanglement", ["The cat sat on the mat."]) == 0.0


def test_answer_relevancy_high_for_matching_text():
    s = answer_relevancy("what is rrf?", "RRF fuses ranked lists.", EMB)
    assert 0.0 < s <= 1.0


def test_faithfulness_proxy_high_when_grounded():
    ctx = ["RRF sums 1/(k+rank) across lists."]
    s = faithfulness_proxy("RRF sums 1/(k+rank) across lists.", ctx, EMB)
    assert s > 0.5


def test_evaluate_question_shape():
    out = evaluate_question("what is rrf?", ["RRF fuses ranked lists."], "RRF fuses ranked lists.", EMB)
    assert set(out) == {"context_keyword_coverage", "answer_relevancy", "faithfulness_proxy"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_ragas_scores_degrades_without_llm():
    assert ragas_scores("q", "ref", ["ctx"], "a", None) == {}


def test_cosine_bounds():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
