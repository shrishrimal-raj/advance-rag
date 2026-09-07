"""Papeer configuration: paths and tunable parameters.

Industry practice: centralize every magic number in one place so tuning the system
(chunk size, k-values, iteration caps) never requires hunting through logic. All paths are
anchored to the project root so the package runs no matter which directory you launch from.
"""
from __future__ import annotations

import sys
import pathlib

# Reach the project root (advance-rag/) so `shared.config` imports resolve.
# This file lives at: <root>/10-capstone-project/code/papeer/config.py
#   parents[0]=papeer  parents[1]=code  parents[2]=10-capstone-project  parents[3]=<root>
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

from shared.config import get_llm, get_llm_provider_name, get_embeddings, get_reranker  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Where sample documents live (course-provided corpus).
DATA_DIR = PROJECT_ROOT / "data" / "samples"
# Optional: drop your own docs here and they'll be ingested too.
USER_DOCS_DIR = PROJECT_ROOT / "data" / "user_docs"

# Persistent vector store + sparse-search sidecar.
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_COLLECTION = "papeer"
CHUNKS_JSON = CHROMA_DIR / "papeer_chunks.json"   # sidecar used by BM25

# --------------------------------------------------------------------------- #
# Retrieval parameters (the knobs you'll tune)
# --------------------------------------------------------------------------- #
DENSE_TOP_K = 10        # how many passages the dense (vector) search returns
BM25_TOP_K = 10         # how many passages the sparse (BM25) search returns
RERANK_TOP_K = 3        # final number of passages handed to the LLM
RRF_K = 60              # Reciprocal Rank Fusion constant (standard value)

# --------------------------------------------------------------------------- #
# Chunking parameters
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 80      # overlap between consecutive chunks (keeps context across boundaries)

# --------------------------------------------------------------------------- #
# Agent parameters
# --------------------------------------------------------------------------- #
MAX_ITERATIONS = 2      # hard cap on the retrieve->...->reflect loop (prevents runaway cost)

# File extensions we know how to ingest.
SUPPORTED_EXTS = {".txt", ".md", ".csv", ".json", ".pdf"}


def supported_files() -> list[pathlib.Path]:
    """Collect every ingestible file from the sample dir + optional user dir."""
    files: list[pathlib.Path] = []
    for d in (DATA_DIR, USER_DOCS_DIR):
        if d.is_dir():
            files.extend(p for p in sorted(d.iterdir()) if p.suffix.lower() in SUPPORTED_EXTS)
    return files
