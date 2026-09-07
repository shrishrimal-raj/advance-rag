"""Pytest suite for papeer.retrieval (Phase 2 — hybrid retrieval + rerank).

Covers:
- known questions return relevant top-k passages (expected keywords present)
- result shape: text/source/score/rank, ranks are 1..k
- metadata filtering narrows results to the requested source/doc_type

No LLM calls. Uses cached local MiniLM embeddings + cached local cross-encoder.
Run:  uv run pytest 10-capstone-project/code/tests/test_retrieval.py -v
"""
from __future__ import annotations

import sys
import pathlib

# Make `papeer` (sibling package) and `shared` (project root) importable.
# This file lives at: <root>/10-capstone-project/code/tests/test_retrieval.py
_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import pytest  # noqa: E402

from papeer.config import CHUNKS_JSON  # noqa: E402
from papeer.ingestion import build_index  # noqa: E402
from papeer.retrieval import retrieve  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_index():
    """Retrieval needs the sidecar + Chroma collection; build once if missing."""
    if not CHUNKS_JSON.exists():
        build_index(verbose=False)
    yield


def _top3_text(question: str) -> str:
    hits = retrieve(question, top_k=3)
    assert hits, f"retrieve() returned no hits for {question!r}"
    return " ".join(h["text"] for h in hits).lower()


# --------------------------------------------------------------------------- #
# Known questions -> relevant top-k
# --------------------------------------------------------------------------- #
def test_falcon_navigation_question_hits_lidar_slam():
    text = _top3_text("What navigation technology does the Falcon AMR use?")
    assert "lidar" in text or "slam" in text, "top-3 must mention Falcon AMR navigation"


def test_hnsw_question_hits_hnsw():
    text = _top3_text("Which vector index builds a multi-layer graph?")
    assert "hnsw" in text, "top-3 must mention HNSW"


def test_emea_products_question_hits_emea():
    text = _top3_text("Which products are available in the EMEA region?")
    assert "emea" in text, "top-3 must mention the EMEA region"


def test_result_shape_and_ranking():
    hits = retrieve("What is RAG?", top_k=3)
    assert len(hits) == 3
    for i, h in enumerate(hits, start=1):
        assert h["rank"] == i, "ranks must be sequential 1..k"
        assert h["text"].strip(), "passage text must be non-empty"
        assert h["source"], "source metadata must be present"
        assert isinstance(h["score"], float), "score must be numeric"


# --------------------------------------------------------------------------- #
# Metadata filtering narrows results
# --------------------------------------------------------------------------- #
def test_metadata_filter_by_source():
    hits = retrieve("products and pricing", top_k=5,
                    metadata_filter={"source": "products.csv"})
    assert hits, "filtered retrieval should still return hits"
    assert all(h["source"] == "products.csv" for h in hits), (
        "every hit must come from products.csv when filtered by source"
    )


def test_metadata_filter_narrows_vs_unfiltered():
    unfiltered = {h["source"] for h in retrieve("products", top_k=5)}
    filtered = retrieve("products", top_k=5, metadata_filter={"doc_type": "CSV"})
    assert all(h["source"] == "products.csv" for h in filtered)
    assert unfiltered - {"products.csv"}, (
        "unfiltered search should span multiple sources; filter must narrow it"
    )
