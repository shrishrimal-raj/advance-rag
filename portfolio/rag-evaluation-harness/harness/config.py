"""Standalone configuration & model factories for the RAG evaluation harness.

Adapted from the course's shared/config.py for a self-contained project:
- Single source of truth for model selection via .env
- 2-tier LLM selection: OpenAI -> Yolo-Auto (OpenAI-compatible) -> None (heuristic mode)
- Local-first embeddings: sentence-transformers MiniLM with a pure-Python
  hashing fallback so the harness ALWAYS runs offline.

NOTE: Ollama is intentionally NOT used — this project must run on a low-power
laptop without a local LLM server. With no API key, pipelines fall back to a
deterministic extractive (heuristic) generator so evaluation stays fully offline.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.0):
    """Return an LLM chain, or None when no cloud key is configured.

    Priority:
    1. OPENAI_API_KEY set    -> OpenAI
    2. YOLO_AUTO_API_KEY set -> Yolo-Auto (OpenAI-compatible, qwen3.8-27b)
    3. otherwise             -> None (caller uses the heuristic generator)
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    yolo_key = os.getenv("YOLO_AUTO_API_KEY", "").strip()
    if yolo_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("YOLO_AUTO_MODEL", "qwen3.8-27b"),
            base_url=os.getenv("YOLO_AUTO_BASE_URL", "https://yolo-auto.com/v1"),
            api_key=yolo_key,
            temperature=temperature,
            max_retries=2,
        )
    return None


def get_llm_provider_name() -> str:
    """Human-readable name of the active LLM provider (for banners/logs)."""
    if os.getenv("OPENAI_API_KEY", "").strip():
        return f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})"
    if os.getenv("YOLO_AUTO_API_KEY", "").strip():
        return f"Yolo-Auto ({os.getenv('YOLO_AUTO_MODEL', 'qwen3.8-27b')})"
    return "heuristic (no LLM key — offline extractive generator)"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (pure Python, zero deps).

    Not semantically strong, but stable and dependency-free — used as the
    guaranteed-offline fallback and in unit tests.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            v[hash(tok) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class SentenceTransformerEmbedder:
    """Thin wrapper around a cached local SentenceTransformer model."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.encode(texts, normalize_embeddings=True)]


def get_embedder(prefer_local_model: bool = True):
    """Return an embedder. Tries the cached MiniLM first, falls back to hashing."""
    if prefer_local_model:
        try:
            return SentenceTransformerEmbedder(
                os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            )
        except Exception:
            pass
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
