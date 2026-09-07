import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enterprise_search.search import SearchService, tokenize  # noqa: E402

DOCS = [
    ("d1", "error 4042 connection pool exhausted retry"),
    ("d2", "how to configure the database connection string"),
    ("d3", "user authentication and password reset flow"),
    ("d4", "connection timeout settings for external services"),
]


def test_tokenize():
    assert tokenize("Error 4042! Connection Pool") == ["error", "4042", "connection", "pool"]


def test_hybrid_ranks_expected_first():
    svc = SearchService(DOCS)
    res = svc.search("connection pool exhausted", top_k=3)
    assert res[0]["id"] == "d1"
    ids = [r["id"] for r in res]
    assert ids.index("d1") < ids.index("d4")


def test_distinct_query_surfaces_other_doc():
    svc = SearchService(DOCS)
    res = svc.search("password reset", top_k=3)
    assert res[0]["id"] == "d3"


def test_top_k_limits_results():
    svc = SearchService(DOCS)
    assert len(svc.search("connection", top_k=2)) <= 2


def test_empty_query_returns_empty():
    svc = SearchService(DOCS)
    assert svc.search("   ", top_k=3) == []


def test_rerank_field_present_and_sorted():
    svc = SearchService(DOCS)
    res = svc.search("connection pool exhausted", top_k=3)
    assert all("rerank" in r for r in res)
    rr = [r["rerank"] for r in res]
    assert rr == sorted(rr, reverse=True)
