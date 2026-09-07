"""FastAPI entrypoint for the Enterprise Knowledge Assistant.

Run:  uv run uvicorn app.main:app --port 8000
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import llm
from .cache import SemanticCache
from .guardrails import sanitize_question
from .retrieval import embed_texts, get_retriever

app = FastAPI(title="Enterprise Knowledge Assistant", version="0.1.0")
_cache: SemanticCache | None = None


def _get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(embed_fn=embed_texts)
    return _cache

SYSTEM = (
    "You are an enterprise knowledge assistant. Answer ONLY from the numbered "
    "context passages below. Cite passages as [n]. If the context is "
    "insufficient, say so explicitly."
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=4, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    r = get_retriever()
    return {"status": "ok", "llm": llm.provider_name(), "docs": r.doc_count}


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    guard = sanitize_question(payload.question)
    if guard.blocked:
        raise HTTPException(status_code=400, detail={"reason": "blocked", "flags": guard.injection_hits})
    question = guard.safe_question or payload.question

    cached = _get_cache().get(question)
    if cached is not None:
        return {"question": question, "answer": cached, "citations": [], "cached": True}

    retriever = get_retriever()
    if retriever.doc_count == 0:
        raise HTTPException(status_code=503, detail="knowledge base empty — ingest documents first")
    docs = retriever.search(question, k=payload.k)
    context = "\n\n".join(f"[{i}] {d.text}" for i, d in enumerate(docs, 1))
    if not llm.is_available():
        # graceful degradation: return top passages so the service stays useful offline
        return {
            "question": question,
            "answer": "LLM unavailable — returning top retrieved passages.\n\n" + context,
            "citations": [d.source for d in docs],
            "cached": False,
            "degraded": True,
        }
    answer = llm.generate(SYSTEM, f"Context:\n{context}\n\nQuestion: {question}")
    _get_cache().put(question, answer)
    return {"question": question, "answer": answer, "citations": [d.source for d in docs], "cached": False}
