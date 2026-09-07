# 🎓 Advanced RAG — Local Course (End-to-End)

> Build production-grade Retrieval-Augmented Generation systems from the ground up.
> Local-first course: every module has **planning → learning → implementation → working code**.

This is a self-paced local recreation of an industry-standard Advanced RAG curriculum
(LangChain + LangGraph + Vector DBs + RAGAS), designed to run **locally with `uv`** and
**no paid API keys required** (OpenAI optional via `.env`).

---

## 🗺️ Course Roadmap

| # | Module | Folder | Key Deliverable | Status |
|---|--------|--------|-----------------|--------|
| 1 | RAG Fundamentals & Architecture | `01-rag-fundamentals/` | Naive RAG pipeline (KB → Retriever → Generator) | ✅ |
| 2 | Document Processing & Chunking | `02-document-processing-chunking/` | Multi-format ingestion + 5 splitter strategies | ✅ |
| 3 | Embeddings & Vector Representations | `03-embeddings-vector-representations/` | Local embedding model + distance metrics lab | ✅ |
| 4 | Vector Stores | `04-vector-stores/` | ChromaDB CRUD + HNSW indexing deep-dive | ✅ |
| 5 | Basic Retrieval Techniques | `05-basic-retrieval-techniques/` | Similarity / Threshold / MMR / Hybrid BM25+Dense / Ensemble | ✅ |
| 6 | Advanced Retrieval Techniques | `06-advanced-retrieval-techniques/` | Compression / Parent-Document / Self-Query / Multi-Query | ✅ |
| 7 | Advanced RAG Patterns | `07-advanced-rag-patterns/` | RAG Fusion+RRF, HyDE, CRAG, Self-RAG, GraphRAG, Multi-Modal | ✅ |
| 8 | Agentic RAG with LangGraph | `08-agentic-rag-langgraph/` | ReAct agent with RAG tool + reflection loop | ✅ |
| 9 | RAG Evaluation with RAGAS | `09-rag-evaluation-ragas/` | Full eval suite: faithfulness, context precision/recall, answer relevancy | ✅ |
| 10 | Capstone Project | `10-capstone-project/` | Production-ready "Papeer" research assistant | ✅ |
| 11 | Production RAG | `11-production-rag/` | Caching, cost optimization, monitoring, guardrails, security | ✅ |

**Suggested order:** strictly sequential (each module builds on the previous).
**Total effort:** ~40 hours of hands-on work.

---

## 🚀 Quick Start (uv)

```bash
# 1. Install uv (if not installed): https://docs.astral.sh/uv/
# 2. From this folder:
uv sync                 # creates .venv + installs all deps
uv run python 01-rag-fundamentals/code/main.py   # run any module
```

### Environment variables (optional but recommended)

```bash
copy .env.example .env    # Windows  (or: cp .env.example .env)
# edit .env → add OPENAI_API_KEY if you want cloud LLMs/embeddings
```

> **No API key?** Every module defaults to **local models**:
> - LLM: instructs you to install [Ollama](https://ollama.com) (`ollama pull llama3.1`)
> - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (auto-downloads, ~90 MB)

### LLM Provider Matrix (3-tier auto-selection in `shared/config.py`)

| Priority | Provider | Env var | Base URL | Default model | Notes |
|---|---|---|---|---|---|
| 1 | OpenAI | `OPENAI_API_KEY` | api.openai.com/v1 | gpt-4o-mini | Cloud default |
| 2 | Yolo-Auto | `YOLO_AUTO_API_KEY` | https://yolo-auto.com/v1 | qwen3.8-27b | OpenAI-compatible, 131k ctx |
| 3 | Ollama (local) | — | localhost:11434 | llama3.1 | Free, needs local GPU/RAM |

### Agent Skills (project guidelines)

| Skill | Path | For |
|---|---|---|
| Cursor rules (always-on) | `.cursor/rules/advance-rag-course.mdc` | Cursor IDE |
| Goose skill | `.agents/skills/advance-rag/SKILL.md` | Goose / Claude-compatible agents |

Both encode the course conventions: Windows UTF-8 guard, shared config imports, graceful degradation, LangChain 1.x API pitfalls, and the verification protocol.

---

## 📁 Module Folder Convention

Every module folder follows the same structure so you can learn one module at a time:

```
NN-module-name/
├── 01-plan.md            # 🎯 Planning: objectives, prerequisites, deliverables, time estimate
├── 02-learning.md        # 📖 Learning: theory, architecture diagrams (mermaid), industry standards
├── 03-implementation.md  # 🔨 Implementation: step-by-step build guide with explanations
├── code/
│   ├── main.py           # ✅ End-to-end working code (runnable: uv run python main.py)
│   └── ...               # supporting scripts / experiments
└── notes.md              # ✍️ YOUR notes (template provided — fill as you go)
```

**Workflow per module:**
1. Read `01-plan.md` → know what you're building and why
2. Read `02-learning.md` → understand concepts + architecture diagrams
3. Follow `03-implementation.md` → build it step by step
4. Run `code/main.py` → verify it works end-to-end
5. Write your takeaways in `notes.md`

---

## 🧰 Tech Stack (Industry Standard)

| Layer | Tool | Why |
|-------|------|-----|
| Package/env | **uv** | Fastest Python package manager; lockfile reproducibility |
| Orchestration | **LangChain + LangGraph** | Industry standard for RAG pipelines & agents |
| Vector DB | **ChromaDB** (local) | Zero-config, HNSW index; swap to Qdrant/pgvector in prod |
| Sparse search | **rank-bm25 / fastembed** | BM25 for hybrid search |
| Embeddings | **sentence-transformers** (local) / OpenAI (optional) | MiniLM = fast local default |
| Reranking | **cross-encoder** (sentence-transformers) | Industry-standard rerank stage |
| Evaluation | **RAGAS** | Standard RAG metrics framework |
| Monitoring | **LangSmith** (optional) | Tracing/debugging in production |
| Guardrails | **NeMo Guardrails / custom validators** | Safety layer for prod RAG |

---

## 📦 Sample Data

`data/samples/` contains documents used across modules (PDF-style text, CSV, JSON, Markdown).
Add your own docs there as you progress.

## 🏆 Production Portfolio

Three standalone, production-grade projects built on top of the course modules (each has its own `pyproject.toml`, tests, Docker, CI, and docs):

| Project | Folder | What it is |
|---------|--------|------------|
| Enterprise Knowledge Assistant | `portfolio/enterprise-knowledge-assistant/` | FastAPI RAG service: hybrid retrieval, metadata filters, citations, Docker + deploy guide |
| Agentic Research Agent | `portfolio/agentic-research-agent/` | LangGraph plan-and-execute + reflection agent with streaming API |
| RAG Evaluation & MLOps Harness | `portfolio/rag-evaluation-harness/` | Offline-first eval pipeline: golden datasets, RAGAS/heuristic metrics, A/B gates, CI quality gate |

Each project runs independently: `cd portfolio/<project> && uv sync && uv run pytest -v`.
## 📝 License / Credits

Local educational project. Curriculum inspired by CampusX "Advanced RAG" course outline.
