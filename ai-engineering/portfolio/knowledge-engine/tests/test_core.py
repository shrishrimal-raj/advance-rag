import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_engine.core import Chunk, KnowledgeEngine, cosine, tokenize  # noqa: E402


def _corpus():
    return [
        Chunk("c1", "PostgreSQL indexes speed up queries", "db-guide", {"topic": "db"}, [1, 0, 0]),
        Chunk("c2", "Redis caching reduces database load", "cache-guide", {"topic": "db"}, [0, 1, 0]),
        Chunk("c3", "React components render UI", "fe-guide", {"topic": "fe"}, [0, 0, 1]),
        Chunk("c4", "Kubernetes orchestrates containers", "ops-guide", {"topic": "ops"}, [0, 0, 0]),
    ]


def test_tokenize():
    assert tokenize("Hello World") == ["hello", "world"]


def test_cosine_orthogonal_and_identical():
    assert abs(cosine([1, 0], [0, 1])) < 1e-9
    assert abs(cosine([1, 1], [1, 1]) - 1.0) < 1e-9


def test_hybrid_retrieval_ranks_relevant_first():
    ke = KnowledgeEngine()
    ke.index(_corpus())
    res = ke.retrieve("database caching redis", query_vector=[0, 1, 0], top_k=3)
    assert res[0]["chunk"].id == "c2"


def test_metadata_filter():
    ke = KnowledgeEngine()
    ke.index(_corpus())
    res = ke.retrieve("anything", filters={"topic": "fe"})
    assert [r["chunk"].id for r in res] == ["c3"]


def test_citations_ordered_with_sources():
    ke = KnowledgeEngine()
    ke.index(_corpus())
    out = ke.answer_context("react ui", top_k=2)
    assert out["citations"][0]["position"] == 1
    assert out["context"] != ""
    assert all(c["source"] for c in out["citations"])


def test_empty_filter_returns_nothing():
    ke = KnowledgeEngine()
    ke.index(_corpus())
    assert ke.retrieve("x", filters={"topic": "nope"}) == []
