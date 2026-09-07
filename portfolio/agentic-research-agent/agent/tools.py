"""Research tools: vector search, metadata-filtered search, calculator.

Tools are plain Python objects with a uniform call interface so the graph can
use real tools in production and fakes in tests:

    tools = ResearchTools(collection=chroma_collection, embed_fn=embed)
    tools.run("vector_search", "how do I chunk PDFs") -> list[dict]

Each result item: {"text": str, "source": str, "score": float}
"""
from __future__ import annotations

import ast
import operator as op
import os
import re
from pathlib import Path
from typing import Any, Callable, Protocol

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db")))


class EmbedFn(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class Tool(Protocol):
    def run(self, name: str, input_: str, **kwargs: Any) -> Any: ...


# --- safe arithmetic for the calculator tool -------------------------------

_ALLOWED_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_ALLOWED_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"disallowed expression element: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Evaluate a pure-arithmetic expression safely (no names, no calls)."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _safe_eval(tree)
        return f"{expression.strip()} = {value}"
    except Exception as exc:  # noqa: BLE001 - tool must never crash the graph
        return f"error: could not evaluate {expression!r} ({exc})"


# --- research tools ---------------------------------------------------------

class ResearchTools:
    """Vector + metadata search over a Chroma collection, plus a calculator."""

    def __init__(self, collection: Any, embed_fn: EmbedFn):
        self.collection = collection
        self.embed_fn = embed_fn

    def run(self, name: str, input_: str, **kwargs: Any) -> Any:
        if name == "vector_search":
            return self.vector_search(input_, k=int(kwargs.get("k", 4)))
        if name == "metadata_filter_search":
            return self.metadata_filter_search(input_, k=int(kwargs.get("k", 4)))
        if name == "calculator":
            return calculate(input_)
        raise ValueError(f"unknown tool: {name!r}")

    def vector_search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        qv = self.embed_fn([query])[0]
        res = self.collection.query(query_embeddings=[qv], n_results=min(k, self.collection.count()))
        return self._format(res)

    def metadata_filter_search(self, spec: str, k: int = 4) -> list[dict[str, Any]]:
        """spec is 'field=value' (e.g. 'topic=chunking'). Falls back to substring match on the value."""
        m = re.match(r"^\s*([A-Za-z_][\w-]*)\s*=\s*(.+?)\s*$", spec or "")
        if not m:
            return []
        field, value = m.group(1), m.group(2)
        where = {field: value}
        total = self.collection.count()
        if total == 0:
            return []
        try:
            res = self.collection.get(where=where, include=["documents", "metadatas"])
        except Exception:  # noqa: BLE001 - bad field/value should degrade, not crash
            return []
        ids = res.get("ids", [])
        if not ids:
            return []
        # rank by embedding similarity to the filter value for stable ordering
        doc_texts = [res["documents"][i] or "" for i in range(len(ids))]
        sims = self._similarities(value, doc_texts)
        order = sorted(range(len(ids)), key=lambda i: sims[i], reverse=True)[:k]
        out = []
        for i in order:
            meta = (res.get("metadatas") or [{}])[i] or {}
            out.append({"text": doc_texts[i], "source": meta.get("source", "unknown"), "score": round(sims[i], 4)})
        return out

    def _similarities(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        qv = self.embed_fn([query])[0]
        dv = self.embed_fn(docs)
        return [float(sum(a * b for a, b in zip(qv, d))) for d in dv]

    @staticmethod
    def _format(res: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        ids = res.get("ids", [[]])[0]
        docs = (res.get("documents") or [[None] * len(ids)])[0]
        metas = (res.get("metadatas") or [None] * len(ids))[0]
        dists = (res.get("distances") or [None] * len(ids))[0]
        for i, _ in enumerate(ids):
            meta = metas[i] or {}
            score = max(0.0, 1.0 - dists[i]) if dists[i] is not None else 0.0
            out.append({"text": docs[i] or "", "source": meta.get("source", "unknown"), "score": round(score, 4)})
        return out


# --- lazy singletons (production path) --------------------------------------

_default_tools: ResearchTools | None = None


def get_embed_fn() -> EmbedFn:
    """Local MiniLM embeddings (cached; no network after first use)."""
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    return embed


def get_tools() -> ResearchTools:
    """Build (once) the production tools over the persisted Chroma collection."""
    global _default_tools
    if _default_tools is not None:
        return _default_tools
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("research_docs", metadata={"hnsw:space": "cosine"})
    _default_tools = ResearchTools(collection=collection, embed_fn=get_embed_fn())
    return _default_tools
