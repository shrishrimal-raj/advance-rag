"""Offline tests for research tools (no LLM, no network)."""
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools import ResearchTools, calculate


@pytest.fixture()
def tools():
    import uuid
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(f"test_{uuid.uuid4().hex[:8]}")
    col.add(
        ids=["a", "b", "c"],
        documents=["Chunking strategies overview", "HNSW index internals", "Embedding model choice"],
        metadatas=[{"topic": "chunking"}, {"topic": "indexing"}, {"topic": "chunking"}],
        embeddings=[[0.1] * 8, [0.2] * 8, [0.3] * 8],
    )
    embed_fn = lambda texts: [[0.1] * 8 for _ in texts]
    return ResearchTools(col, embed_fn)


def test_calculate_basic():
    assert calculate("2+3*4") == "2+3*4 = 14"


def test_calculate_rejects_unsafe():
    out = calculate("__import__('os').system('id')")
    assert out.startswith("error:")


def test_run_dispatch_calculator(tools):
    assert "= 2.5" in tools.run("calculator", "10/4")


def test_run_unknown_tool_raises(tools):
    with pytest.raises(ValueError):
        tools.run("bogus_tool", "x")


def test_vector_search_returns_docs(tools):
    docs = tools.vector_search("chunking", k=2)
    assert len(docs) == 2
    assert all(d["text"] and d["source"] for d in docs)


def test_metadata_filter_narrows(tools):
    docs = tools.metadata_filter_search("topic=chunking", k=5)
    assert len(docs) == 2
    assert all("Chunking" in d["text"] or "Embedding" in d["text"] for d in docs)


def test_metadata_filter_bad_spec_returns_empty(tools):
    assert tools.metadata_filter_search("no equals sign here") == []
