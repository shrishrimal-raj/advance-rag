"""Pytest suite for papeer.ingestion (Phase 1 — multi-format ingestion).

Covers:
- format-aware loaders return non-empty text for every sample document
- stable chunk IDs are deterministic (the idempotency mechanism)
- a full build_index() run yields non-empty chunks, each carrying citation metadata
- re-ingestion is idempotent: same chunk IDs, no duplicates

No LLM calls. Embeddings are the cached local MiniLM model only.
Run:  uv run pytest 10-capstone-project/code/tests/test_ingestion.py -v
"""
from __future__ import annotations

import sys
import pathlib

# Make `papeer` (sibling package) and `shared` (project root) importable.
# This file lives at: <root>/10-capstone-project/code/tests/test_ingestion.py
_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import pytest  # noqa: E402

from papeer import ingestion  # noqa: E402
from papeer.config import CHUNKS_JSON, DATA_DIR  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_files() -> list[pathlib.Path]:
    files = sorted(DATA_DIR.iterdir())
    assert files, f"expected sample documents in {DATA_DIR}"
    return files


@pytest.fixture(scope="module")
def first_build_ids() -> set[str]:
    """Run ingestion once; return the chunk-ID set it produced."""
    ingestion.build_index(verbose=False)
    assert CHUNKS_JSON.exists(), "ingestion must write the BM25 sidecar"
    import json
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    assert chunks, "ingestion must produce at least one chunk"
    return {c["id"] for c in chunks}


# --------------------------------------------------------------------------- #
# Loaders & ID scheme (fast, no embeddings)
# --------------------------------------------------------------------------- #
def test_supported_files_found(sample_files):
    from papeer.config import supported_files
    found = supported_files()
    assert len(found) >= 4, "all four sample docs (txt/md/csv/json) must be ingestible"


def test_load_file_returns_nonempty_text(sample_files):
    for path in sample_files:
        blocks = ingestion._load_file(path)
        assert blocks, f"loader returned no blocks for {path.name}"
        assert any(b.strip() for b in blocks), f"all blocks empty for {path.name}"


def test_stable_id_is_deterministic():
    a = ingestion._stable_id("products.csv", "1-0")
    b = ingestion._stable_id("products.csv", "1-0")
    c = ingestion._stable_id("products.csv", "1-1")
    assert a == b, "same (source, index) must always map to the same id"
    assert a != c, "different chunk indices must map to different ids"
    assert isinstance(a, str) and len(a) > 0


# --------------------------------------------------------------------------- #
# Full build_index() run
# --------------------------------------------------------------------------- #
def test_build_index_yields_chunks_with_metadata(first_build_ids):
    import json
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))

    assert len(chunks) > 0, "expected non-empty chunk list"
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk IDs must be unique within one build"

    required_meta = ("source", "page", "chunk_index", "doc_type")
    for c in chunks:
        assert c["text"].strip(), "every chunk must have non-empty text"
        for key in required_meta:
            assert key in c, f"chunk {c['id']} missing metadata key {key!r}"
        assert c["source"], "source metadata must be non-empty"
        assert c["doc_type"] in {"TXT", "MD", "CSV", "JSON", "PDF"}


def test_reingestion_is_idempotent(first_build_ids):
    """Re-running ingestion must reproduce the exact same ID set — no duplicate vectors."""
    import json
    ingestion.build_index(verbose=False)
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    second_ids = [c["id"] for c in chunks]

    assert len(second_ids) == len(set(second_ids)), "no duplicate IDs after re-ingestion"
    assert set(second_ids) == first_build_ids, (
        "re-ingestion must be idempotent: identical corpus => identical chunk IDs"
    )
