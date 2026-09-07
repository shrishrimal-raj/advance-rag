import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enterprise_search.core import Doc, EnterpriseSearchEngine, cosine, tokenize  # noqa: E402


def test_tokenize():
    assert tokenize("A B") == ["a", "b"]


def test_cosine_orthogonal():
    assert abs(cosine([1, 0], [0, 1])) < 1e-9


def test_tenant_isolation_no_cross_leak():
    e = EnterpriseSearchEngine()
    e.upsert("acme", [Doc("a1", "acme secret internal doc", "acme-wiki", {"type": "wiki"}, [1, 0])])
    e.upsert("globex", [Doc("g1", "globex public doc", "globex-wiki", {"type": "wiki"}, [0, 1])])
    res = e.search("acme", "secret internal doc")
    assert all(r["doc"].id.startswith("a") for r in res)
    assert "g1" not in [r["doc"].id for r in res]


def test_unknown_tenant_returns_empty():
    e = EnterpriseSearchEngine()
    e.upsert("acme", [Doc("a1", "hello world", "src", {}, [1, 0])])
    assert e.search("nobody", "hello world") == []


def test_hybrid_ranks_relevant_first():
    # >=4 docs so BM25 IDF discriminates (tiny corpora give idf=log(1)=0).
    e = EnterpriseSearchEngine()
    e.upsert("t", [
        Doc("d1", "kubernetes pod scheduling and autoscaling", "ops", {"team": "infra"}, [1, 0]),
        Doc("d2", "postgres replication lag monitoring", "db", {"team": "data"}, [0, 1]),
        Doc("d3", "react state management with hooks", "fe", {"team": "web"}, [0, 0]),
        Doc("d4", "docker image layer caching tips", "ops", {"team": "infra"}, [0, 0]),
        Doc("d5", "terraform infrastructure as code", "ops", {"team": "infra"}, [0, 0]),
        Doc("d6", "postgres connection pooling with pgbouncer", "db", {"team": "data"}, [0, 0.9]),
    ])
    res = e.search("t", "postgres replication", query_vector=[0, 1], top_k=2)
    assert res[0]["doc"].id == "d2"


def test_metadata_filter_within_tenant():
    e = EnterpriseSearchEngine()
    e.upsert("t", [
        Doc("d1", "alpha beta", "s", {"team": "infra"}, [1, 0]),
        Doc("d2", "alpha gamma", "s", {"team": "data"}, [0, 1]),
    ])
    res = e.search("t", "alpha", filters={"team": "data"})
    assert [r["doc"].id for r in res] == ["d2"]


def test_citations_present():
    e = EnterpriseSearchEngine()
    e.upsert("t", [Doc("d1", "x y z", "src", {}, [1, 0])])
    res = e.search("t", "x y z")
    assert res[0]["citation"]["position"] == 1
    assert res[0]["citation"]["source"] == "src"
