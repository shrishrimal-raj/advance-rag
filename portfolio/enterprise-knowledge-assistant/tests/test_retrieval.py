"""Offline tests for hybrid retrieval (no LLM, no network)."""
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval import HybridRetriever, chunk_markdown


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    data = tmp_path_factory.mktemp("kb")
    (data / "lidar.md").write_text(
        "Lidar SLAM builds maps from laser scans. Odometry fuses wheel and IMU data.", encoding="utf-8"
    )
    (data / "hnsw.md").write_text(
        "HNSW indexes vectors with hierarchical navigable small world graphs for fast ANN search.", encoding="utf-8"
    )
    (data / "emea.md").write_text(
        "EMEA product launches require regional compliance review before go-to-market.", encoding="utf-8"
    )
    r = HybridRetriever(persist_dir=data / "chroma", collection="test_kb")
    n = r.ingest_files(data)
    assert n > 0
    return r


def test_chunk_markdown_nonempty():
    chunks = chunk_markdown("# Title\n\nSome body text here that is long enough to chunk properly.\n")
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_rrf_fuse_top_score_is_one(retriever):
    # Both lists agree on rank 1 -> normalized score must be exactly 1.0
    fused = retriever._rrf_fuse([(0, 0.9), (1, 0.5)], [(0, 0.8), (2, 0.4)])
    assert fused[0] == pytest.approx(1.0)


def test_rrf_fuse_disjoint_scores_below_one(retriever):
    fused = retriever._rrf_fuse([(0, 0.9)], [(1, 0.8)])
    assert 0.0 < fused[0] < 1.0
    assert 0.0 < fused[1] < 1.0


def test_search_returns_ranked_docs(retriever):
    docs = retriever.search("how does HNSW speed up vector search?", k=3)
    assert 1 <= len(docs) <= 3
    top = docs[0]
    assert top.text and top.source
    assert 0.0 < top.score <= 1.0
    scores = [d.score for d in docs]
    assert scores == sorted(scores, reverse=True)


def test_search_hits_expected_doc(retriever):
    docs = retriever.search("EMEA compliance review", k=2)
    assert any("EMEA" in d.text for d in docs)
