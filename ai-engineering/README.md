# 🚀 SDE → AI Engineer — Local Course (10-Week Cohort Recreation)

> Build production-grade **AI systems** from the ground up: LLM fundamentals → RAG → agents →
> evals → MCP → fine-tuning → system design → multimodal → capstone.
> Local-first course: every week has **planning → learning → implementation → working code**.

A self-paced local recreation of the Coding Shuttle "From SDE To AI Engineer" 10-week cohort
(curriculum in `Task.md`), designed to run **locally with `uv`** and use a **cloud LLM**
(Yolo-Auto) so it works on an 8 GB laptop with no local model serving.

---

## 🗺️ Course Roadmap

| # | Week | Folder | Weekly Build | Status |
|---|------|--------|--------------|--------|
| 1 | Python for AI & LLM Fundamentals | `01-python-llm-fundamentals/` | The AI Gateway (streaming chat backend + cost tracking) | 🔄 |
| 2 | RAG Foundations | `02-rag-foundations/` | The Knowledge Engine (context-aware RAG) | ⏳ |
| 3 | Enterprise-grade RAG Pipelines | `03-enterprise-rag-pipelines/` | The Enterprise Search Engine (hybrid + rerank) | ⏳ |
| 4 | AI Agents & State Machines | `04-ai-agents-state-machines/` | The Persistent Operator (tool-calling + Redis memory) | ⏳ |
| 5 | Evals & Observability | `05-evals-observability/` | The Regression & Telemetry Gate (CI-gated evals) | ⏳ |
| 6 | MCP, Context Eng & Multi-Agent | `06-mcp-context-multi-agent/` | The Audited Tool Bridge (MCP servers) | ⏳ |
| 7 | ML Fundamentals & Fine-Tuning | `07-ml-fine-tuning/` | The Specialist Model (LoRA/QLoRA + benchmark) | ⏳ |
| 8 | Agentic System Design & Reliability | `08-agentic-system-design/` | The Dual-Agent Supervisor (failover drill) | ⏳ |
| 9 | Multimodal AI & Capstone Kickoff | `09-multimodal-ai/` | The Multimodal RAG Engine (vision + voice) | ⏳ |
| 10 | Capstone Project: Finale & Demo Day | `10-capstone-finale/` | The Finale Deploy (enterprise MVP) | ⏳ |

**Suggested order:** strictly sequential (each week builds on the previous).
**Progress:** see `CHECKPOINT.md` (live tracker).

---

## 🚀 Quick Start (uv)

```bash
cd ai-engineering
uv sync                 # creates .venv + installs deps
uv run python 01-python-llm-fundamentals/code/main.py   # run any week
```

### Environment variables
```bash
cp ../.env ./.env       # reuse the repo-root .env (has YOLO_AUTO_API_KEY)
# or copy .env.example and fill in your own keys
```

> **No API key?** Scripts degrade gracefully with a clear hint (exit 0). For full runs set
> `YOLO_AUTO_API_KEY` (cloud, works on 8 GB) — see provider matrix below.

### LLM Provider Matrix (3-tier auto-selection in `shared/config.py`)

| Priority | Provider | Env var | Base URL | Default model | Notes |
|---|---|---|---|---|---|
| 1 | OpenAI | `OPENAI_API_KEY` | api.openai.com/v1 | gpt-4o-mini | Cloud default |
| 2 | Yolo-Auto | `YOLO_AUTO_API_KEY` | https://yolo-auto.com/v1 | qwen3.8-27b | OpenAI-compatible, 131k ctx (**used here**) |
| 3 | Ollama (local) | — | localhost:11434 | llama3.1 | Free, needs local GPU/RAM (**not on this laptop**) |

---

## 📁 Week Folder Convention

Every week folder follows the same structure:

```
NN-week-name/
├── 01-plan.md            # 🎯 Planning: objectives, prerequisites, deliverables, time estimate
├── 02-learning.md        # 📖 Learning: theory + architecture diagrams (mermaid, noob→expert)
├── 03-implementation.md  # 🔨 Implementation: step-by-step build guide
├── code/
│   ├── main.py           # ✅ End-to-end working code (framework-based)
│   ├── approach_2_*.py   # 🔄 Alternative framework usage
│   ├── approach_3_*.py   # 🧠 From-scratch implementation (proves deep understanding)
│   └── *_benchmark.py    # 📊 Head-to-head comparison where meaningful
└── notes.md              # ✍️ YOUR notes (template provided)
```

**Workflow per week:** read plan → read learning (+ diagrams) → follow implementation → run code → take notes.

---

## 🧰 Tech Stack (Industry Standard)

| Layer | Tool | Why |
|-------|------|-----|
| Package/env | **uv** | Fastest Python package manager; lockfile reproducibility |
| Web/API | **FastAPI** + uvicorn + httpx | Streaming non-blocking backends |
| Orchestration | **LangChain + LangGraph** | Industry standard for RAG pipelines & agents |
| Vector DB | **ChromaDB** (local) / pgvector (prod) | Zero-config HNSW; swap to pgvector/Qdrant in prod |
| Sparse search | **rank-bm25 / fastembed** | BM25 for hybrid search |
| Embeddings | **sentence-transformers** (local) / cloud | MiniLM = fast local default |
| Reranking | **cross-encoder** (sentence-transformers) | Industry-standard rerank stage |
| Agents/Memory | **LangGraph + Redis** | State machines + persistent memory |
| Tools/Protocol | **MCP (FastMCP)** | Standardize tools for agents |
| Multi-agent | **CrewAI / LangGraph supervisor** | Orchestration patterns |
| Evaluation | **RAGAS / DeepEval** | Standard RAG metrics |
| Observability | **Langfuse / OpenTelemetry** | Tracing + unit economics |
| Fine-tuning | **PEFT (LoRA/QLoRA)** | Parameter-efficient fine-tuning (cloud/GPU) |
| UI/Dashboards | **Streamlit** | Live dashboards |
| Deployment | **Docker** | Containerized deployment |

---

## 🏆 Production Portfolio

Three standalone, production-grade projects built on top of the course weeks (each has its own
`pyproject.toml`, tests, Docker, CI, and docs):

| Project | Folder | What it is |
|---------|--------|------------|
| AI Gateway Service | `portfolio/ai-gateway-service/` | FastAPI streaming LLM service: cost tracking, Docker + CI + monitoring |
| Enterprise Search Service | `portfolio/enterprise-search-service/` | Multi-tenant hybrid search API: BM25+dense+rerank, citations |
| Regression & Telemetry Gate | `portfolio/regression-telemetry-gate/` | CI-gated eval harness: golden datasets, A/B gates, tracing |

Each project runs independently: `cd portfolio/<project> && uv sync && uv run pytest -v`.

---

## 📝 License / Credits

Local educational project. Curriculum inspired by Coding Shuttle "From SDE To AI Engineer" cohort outline (`Task.md`).
