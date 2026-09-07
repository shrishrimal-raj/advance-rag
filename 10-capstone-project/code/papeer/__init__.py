"""Papeer — a production-ready research-assistant RAG system (Module 10 capstone).

Package layout:
    config.py     paths & tunable parameters
    ingestion.py  multi-format ingestion -> Chroma + BM25 sidecar
    retrieval.py  hybrid (dense + BM25) -> RRF -> cross-encoder rerank
    agent.py      LangGraph agent: retrieve -> grade -> generate -> reflect
    api.py        FastAPI service (POST /ask)  [needs: uv add fastapi uvicorn]
    evaluate.py   RAGAS evaluation on a built-in golden set
    main.py       CLI demo: 3 questions through the full pipeline w/ timing
"""
__version__ = "1.0.0"
