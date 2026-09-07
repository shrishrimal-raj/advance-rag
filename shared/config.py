"""Shared configuration & model factories for the Advanced RAG course.

Usage in any module:
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from shared.config import get_llm, get_embeddings, CHROMA_DIR

Design (industry standard):
- Single source of truth for model selection via .env
- 3-tier LLM selection: OpenAI -> Yolo-Auto (OpenAI-compatible) -> Ollama (local)
- Local-first: works with zero API keys (Ollama LLM + sentence-transformers embeddings)
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (works no matter which module folder you run from)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_llm(temperature: float = 0.0):
    """Return an LLM chain.

    Priority:
    1. OPENAI_API_KEY set     -> OpenAI
    2. YOLO_AUTO_API_KEY set  -> Yolo-Auto (OpenAI-compatible, qwen3.8-27b, 131k ctx)
    3. otherwise              -> Ollama (local)

    Ollama must be running:  ollama serve   (and: ollama pull llama3.1)
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

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )


def get_llm_provider_name() -> str:
    """Human-readable name of the active LLM provider (for banners/logs)."""
    if os.getenv("OPENAI_API_KEY", "").strip():
        return f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})"
    if os.getenv("YOLO_AUTO_API_KEY", "").strip():
        return f"Yolo-Auto ({os.getenv('YOLO_AUTO_MODEL', 'qwen3.8-27b')})"
    return f"Ollama local ({os.getenv('OLLAMA_MODEL', 'llama3.1')})"


def get_embeddings():
    """Return an embedding model.

    Default: local sentence-transformers MiniLM (384 dims, fast, free).
    Set EMBEDDING_PROVIDER=openai in .env to use text-embedding-3-small.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )


def get_reranker():
    """Cross-encoder reranker (local, industry-standard rerank stage)."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
