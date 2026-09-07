---
name: advance-rag
description: Expert workflow for building, extending, debugging, and verifying the Advanced RAG course (11 modules) and its production portfolio projects. Use when adding module examples, fixing LangChain/ChromaDB bugs, creating portfolio projects, writing RAG documentation, or verifying code runs.
---

# Advanced RAG Course — Expert Workflow Skill

This project is a locally-runnable Advanced RAG course (uv + LangChain 1.x + LangGraph +
ChromaDB + RAGAS) with production-grade portfolio projects under `portfolio/`.

## When to use
- Adding new examples / implementation approaches to a module
- Fixing runtime errors in any module or portfolio project
- Creating a new portfolio project or deployment artifact
- Writing or upgrading RAG documentation (must include mermaid diagrams)

## Project map
- `shared/config.py` — ONLY place that constructs LLMs/embeddings. 3-tier: OpenAI → Yolo-Auto (`https://yolo-auto.com/v1`, model `qwen3.8-27b`, env `YOLO_AUTO_API_KEY`) → Ollama local.
- `01-…` through `11-…` — course modules. Each: `01-plan.md`, `02-learning.md`, `03-implementation.md`, `code/`, `notes.md`.
- `portfolio/` — standalone production projects (own pyproject, tests, Docker, CI).
- `MASTER_PLAN.md` — deliverable tracker; update its status table when work completes.
- `data/samples/` — shared sample documents (txt/csv/json/md).

## Mandatory conventions
1. Run everything with `uv run python ...` (never bare python).
2. Every runnable script starts with:
   ```python
   import sys
   if sys.platform == "win32":
       sys.stdout.reconfigure(encoding="utf-8", errors="replace")
       sys.stderr.reconfigure(encoding="utf-8", errors="replace")
   ```
3. Shared imports: `sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))`
   (`parents[2]` for scripts inside `code/<subdir>/`), then `from shared.config import get_llm, get_embeddings`.
4. Graceful degradation: missing Ollama/cloud key ⇒ clear hint + exit 0, never a traceback.
5. Pretty output via `rich`.

## Known API pitfalls (LangChain 1.x era)
- `rank_bm25.BM25Okapi` (renamed from `RankBM25`).
- Advanced retrievers live in `langchain_classic.retrievers` (fallback: `langchain.retrievers`).
- `LLMChainExtractor.from_llm(llm)`.
- `ParentDocumentRetriever`: choose `byte_store` vs `parent_doc_store` via `model_fields` inspection.
- `SelfQueryRetriever.from_llm(..., document_contents=..., metadata_field_info=[...])`; strip literal braces from sample text.
- `InMemoryStore` from `langchain_core.stores`.
- `JSONLoader` requires `jq_schema='.'`.
- `MarkdownHeaderTextSplitter` has no `split_documents` in the new API — use `split_text`.
- Custom ChromaDB embedding-function adapters need both `__call__` and `embed_query` (list-aware).
- `langchain-community` must stay `<0.4` (0.4 removed symbols `langchain_classic` imports).

## Verification protocol (ALWAYS, before reporting done)
1. `uv run python -m py_compile` every new/changed file.
2. Fully execute scripts that need no LLM and no >100MB downloads.
3. Never execute scripts that download large models (CLIP etc.) — compile-check only.
4. State exactly which commands ran and their outcomes.

## Documentation standards
- Mermaid diagrams for every architecture explanation; progress noob → practitioner → expert.
- Trade-off tables (cost / quality / latency) for every design choice.
- Portfolio projects need: README, `docs/PLANNING.md`, `docs/DESIGN.md`, `docs/DEPLOYMENT.md`,
  `tests/`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `.env.example`.

## Secrets
Never commit real keys. `.env` is gitignored; only `.env.example` (empty values) is tracked.
