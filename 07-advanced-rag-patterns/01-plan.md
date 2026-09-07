# Module 07: Advanced RAG Patterns — Plan

## Objectives

By the end of this module you will be able to:

1. **Diagnose** why a basic "retrieve-then-generate" RAG pipeline fails on a given query.
2. **Select** the right advanced pattern (RAG Fusion, HyDE, CRAG, Self-RAG, GraphRAG, Multi-Modal RAG) for a specific failure mode.
3. **Implement** each pattern end-to-end in Python using LangChain + Chroma + local Ollama.
4. **Evaluate** trade-offs: extra LLM calls, latency, accuracy gains, and infrastructure cost.
5. **Combine** patterns (e.g., RAG Fusion + CRAG validation) for production-grade systems.

## Prerequisites

| Topic | Where covered |
|-------|---------------|
| Basic RAG pipeline (chunk → embed → retrieve → generate) | Modules 01–04 |
| Chroma vector store basics | Module 04 |
| Ollama / OpenAI LLM integration | `shared/config.py` |
| Sentence-transformers embeddings | Module 03 |
| Python: dicts, BFS, BeautifulSoup | General Python |

## Per-Pattern Deliverables Checklist

### 1. RAG Fusion + Reciprocal Rank Fusion
- [ ] LLM generates 3 query variants from a single user question
- [ ] Retrieve top-5 per variant from in-memory Chroma
- [ ] Implement RRF (k=60) inline — no external library
- [ ] Print fused ranking with per-query rank columns
- [ ] Generate final answer from fused context

### 2. HyDE (Hypothetical Document Embeddings)
- [ ] LLM writes a hypothetical answer document
- [ ] Embed the hypothetical doc (NOT the user query)
- [ ] Retrieve using the hypothetical doc as query vector
- [ ] Generate real grounded answer from retrieved chunks
- [ ] Output explains WHY this helps when vocab differs

### 3. CRAG (Corrective RAG)
- [ ] Retrieve top-3 documents
- [ ] LLM "knowledge validator" classifies each: Correct / Incorrect / Ambiguous
- [ ] If good → answer from docs
- [ ] If bad → web fallback via DuckDuckGo HTML (requests + BeautifulSoup)
- [ ] On web failure → degrade to "insufficient knowledge" response
- [ ] Print validation decisions clearly

### 4. Self-RAG
- [ ] Generate initial answer with citations
- [ ] REFLECTION pass: LLM critiques answer vs evidence (relevant? supported? complete?)
- [ ] If critique fails → regenerate ONCE with critique fed back
- [ ] Print reflection decisions and final answer

### 5. GraphRAG (Lightweight)
- [ ] LLM extracts entities + relations from corpus chunks → dict-of-dicts graph
- [ ] Communities via connected components (pure Python BFS)
- [ ] LLM summarizes each community
- [ ] Answer a GLOBAL question using community summaries (not raw chunks)
- [ ] Print graph nodes/edges, communities, summaries, final answer

### 6. Multi-Modal RAG
- [ ] Create 2–3 labeled images at runtime with PIL (ImageDraw)
- [ ] Embed images with CLIP (`clip-ViT-B-32`) inside try/except
- [ ] On ANY failure → fall back to text-only demo explaining cross-modal architecture
- [ ] Store image embeddings alongside text chunks in Chroma
- [ ] Cross-modal queries: text→image and image→text
- [ ] Comment noting production uses larger CLIP/SigLIP models

## Time Estimate

| Section | Time |
|---------|------|
| Reading theory (02-learning.md) | 90 min |
| RAG Fusion implementation | 45 min |
| HyDE implementation | 40 min |
| CRAG implementation | 50 min |
| Self-RAG implementation | 45 min |
| GraphRAG implementation | 55 min |
| Multi-Modal RAG implementation | 45 min |
| Review + experiments | 30 min |
| **Total** | **~6 hours** |

## Success Criteria

You have completed this module when you can:

1. Run all 6 scripts successfully: `uv run python 07-advanced-rag-patterns/code/<pattern>/main.py`
2. Explain in your own words what problem each pattern solves that basic RAG cannot.
3. Given a new scenario (e.g., "user asks about a topic not in the corpus"), pick the right pattern and justify why.
4. Modify at least one script (e.g., change RRF k-value, add a 4th query variant, swap the validator prompt) and observe the effect.
5. Fill in `notes.md` with your key takeaways and at least one experiment you tried.
