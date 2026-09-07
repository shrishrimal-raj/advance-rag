"""Environment-driven settings for the Enterprise Knowledge Assistant.

Single source of truth: every tunable is read from the environment (or the
project-local .env file). No hard-coded secrets, no global mutation at import
time beyond loading dotenv.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # --- LLM (OpenAI-compatible; Yolo-Auto is the default cloud provider) ---
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    yolo_auto_api_key: str = field(default_factory=lambda: _env("YOLO_AUTO_API_KEY"))
    yolo_auto_base_url: str = field(
        default_factory=lambda: _env("YOLO_AUTO_BASE_URL", "https://yolo-auto.com/v1")
    )
    yolo_auto_model: str = field(default_factory=lambda: _env("YOLO_AUTO_MODEL", "qwen3.8-27b"))

    # --- Vector store ---
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(_env("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db")))
    )
    collection_name: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "knowledge"))

    # --- Embeddings (local MiniLM by default: free, cached, fast) ---
    embedding_model: str = field(
        default_factory=lambda: _env("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))

    # --- Retrieval ---
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("RERANK_ENABLED", False))
    rerank_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # --- Semantic cache ---
    cache_enabled: bool = field(default_factory=lambda: _env_bool("CACHE_ENABLED", True))
    cache_threshold: float = field(default_factory=lambda: _env_float("CACHE_THRESHOLD", 0.92))

    # --- Guardrails ---
    guardrails_enabled: bool = field(default_factory=lambda: _env_bool("GUARDRAILS_ENABLED", True))

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key or self.yolo_auto_api_key)


settings = Settings()
