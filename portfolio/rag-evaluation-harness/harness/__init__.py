"""RAG Evaluation & MLOps Harness.

Standalone harness for evaluating RAG pipelines:
- custom OFFLINE metrics (no API keys required)
- optional RAGAS metrics (gracefully skipped without an LLM key)
- A/B pipeline comparison (naive dense vs hybrid BM25+dense+RRF+rerank)
- regression gate that blocks quality drops in CI
"""

__version__ = "0.1.0"
