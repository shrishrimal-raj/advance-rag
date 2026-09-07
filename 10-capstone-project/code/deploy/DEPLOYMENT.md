# 🚀 Papeer — Production Deployment Runbook

Containerized deployment of the Papeer capstone RAG assistant (Streamlit UI over the
`papeer` package: hybrid retrieval → cross-encoder rerank → LLM answer).

```
10-capstone-project/code/deploy/
├── Dockerfile          # python:3.12-slim + uv, non-root, healthcheck
├── docker-compose.yml  # app service + persistent Chroma volume
└── DEPLOYMENT.md       # this runbook
```

---

## 1. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| Docker Engine ≥ 24 + Compose v2 | build & orchestration | `docker compose version` |
| Repo-root `.env` with an LLM key | answer generation | `test -n "$YOLO_AUTO_API_KEY" -o -n "$OPENAI_API_KEY"` |
| ~2 GB free disk | image + HF model cache | `df -h .` |
| 4 GB RAM headroom | MiniLM embeddings + cross-encoder load into memory | — |

**First build is slow on purpose:** `uv sync --frozen --no-dev` installs locked deps, and the
first query downloads the two small local models (MiniLM-L6-v2 embeddings ≈ 90 MB,
cross-encoder ms-marco-MiniLM-L-6-v2 ≈ 90 MB) into the image's `HF_HOME`. Subsequent
containers reuse the layer cache; in production you should pre-warm or mount a shared
HF cache volume (see §7).

## 2. Build & Run

```bash
# From the repo root:
docker build -f 10-capstone-project/code/deploy/Dockerfile -t papeer .

# Option A — plain docker:
docker run --rm -p 8501:8501 --env-file .env \
  -v papeer-chroma:/app/data/chroma_db papeer

# Option B — compose (recommended; named volume + healthcheck + restart policy):
cd 10-capstone-project/code/deploy
docker compose up --build -d
docker compose ps          # wait for "healthy"
open http://localhost:8501
```

The UI ships **without a pre-built index**: use the sidebar → *Rebuild index* button (or run
ingestion once inside the container) so the first user doesn't pay embedding latency.

### Smoke test

```bash
curl -fsS http://localhost:8501/_stcore/health   # -> "ok"
docker compose logs app | tail -20
```

## 3. Environment Variables

All secrets are injected via `--env-file .env` / compose `env_file` — **never baked into the image**.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `YOLO_AUTO_API_KEY` | one of the LLM keys | — | Yolo-Auto cloud LLM (OpenAI-compatible) |
| `YOLO_AUTO_BASE_URL` | no | `https://yolo-auto.com/v1` | Yolo-Auto endpoint |
| `YOLO_AUTO_MODEL` | no | `qwen3.8-27b` | Yolo-Auto model id |
| `OPENAI_API_KEY` | one of the LLM keys | — | OpenAI LLM (takes priority if set) |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | OpenAI model id |
| `EMBEDDING_PROVIDER` | no | `local` | `local` (MiniLM, cached) or `openai` |
| `LOCAL_EMBEDDING_MODEL` | no | `sentence-transformers/all-MiniLM-L6-v2` | local embedding model |
| `OLLAMA_MODEL` / `OLLAMA_BASE_URL` | only if no cloud key | `llama3.1` / `http://localhost:11434` | last-resort local LLM (needs Ollama sidecar) |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | no | off | LangSmith tracing (§8) |

Provider precedence (see `shared/config.py`): `OPENAI_API_KEY` → `YOLO_AUTO_API_KEY` → Ollama.
If none is reachable the UI still works in **retrieval-only mode** and shows a hint banner —
it never crashes.

## 4. Scaling Notes

- **Stateless workers:** the app container is stateless *except* the Chroma dir. With the
  named volume you get **one** healthy replica (Chroma's persistent client is single-node).
  To scale horizontally: move the index to an external store (Qdrant/pgvector/Milvus), keep
  workers fully stateless, and put an LB in front. That is the standard path from "works on
  one box" to "survives a deploy".
- **Ingestion stays offline.** Never embed documents inside a request path. In production,
  run `build_index()` as a scheduled job/worker that watches the document corpus and rebuilds
  the collection (it is idempotent — stable chunk IDs make re-runs safe).
- **Memory:** each worker holds MiniLM + cross-encoder (~1–1.5 GB). The compose file caps the
  container at 4 GB; size replicas accordingly.
- **CPU-bound rerank:** the cross-encoder scores ~20 candidates per query; on a low-power CPU
  that's the main per-query cost after the LLM call. Batch queries or pre-compute for known
  questions if latency matters.

## 5. Backup & Restore (vector store)

Everything the system persists lives under `data/chroma_db/` (container path
`/app/data/chroma_db`): `chroma.sqlite3` (metadata + HNSW segments) plus the
`papeer_chunks.json` BM25 sidecar.

```bash
# Backup (compose): stop writes, tar the volume, resume.
docker compose stop app
docker run --rm -v papeer-chroma:/data -v "$(pwd)/backups:/backup" alpine \
  tar czf /backup/papeer-$(date +%F).tar.gz -C /data .
docker compose start app

# Restore:
docker compose stop app
docker run --rm -v papeer-chroma:/data -v "$(pwd)/backups:/backup" alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/papeer-YYYY-MM-DD.tar.gz -C /data"
docker compose start app
```

**Golden rule:** the index is *derived data*. If the backup is ever lost, re-running
`build_index()` over the source corpus reproduces it exactly (same chunk IDs, same
embeddings — as long as the embedding model version is unchanged). Back up the **source
documents**, not just the index.

⚠️ Embeddings are not portable across models: changing `LOCAL_EMBEDDING_MODEL` or
`EMBEDDING_PROVIDER` silently invalidates the whole index — always rebuild after such a change.

## 6. Monitoring Hooks

- **Health:** the container healthcheck hits Streamlit's `/_stcore/health`; wire it to your
  orchestrator's alerting (compose `restart: unless-stopped` already self-heals crashes).
- **Latency/cost metrics:** every agent stage records wall-clock time (`agent.py` →
  `state['stages']`). Export these to your metrics backend (Prometheus `/metrics` endpoint,
  Datadog, etc.) and alert on p95 end-to-end latency and per-query token spend.
- **Feedback loop:** add 👍/👎 to the UI and log `(question, answer_id, verdict)` — this is
  the raw material for the RAGAS eval gate (Module 09).
- **Log hygiene:** structured logs with a request ID correlating retrieve → rerank → generate
  stages (Module 11 covers the full observability stack).

## 7. LangSmith Tracing (optional but recommended)

LangSmith gives a visual trace tree per request (each LangGraph node, token counts, latencies):

1. Create a project at https://smith.langchain.com and copy the API key.
2. Add to `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_...
   LANGCHAIN_PROJECT=papeer
   ```
3. `docker compose up -d` — every `/ask`-style run now appears in the project's trace view.
   Sample (don't trace 100%) in high-traffic deployments to control cost/PII exposure.

## 8. Rollback Procedure

1. **Tag images by git SHA** when building (`-t papeer:<sha>`); never rely on `latest` alone.
2. Keep the previous good image + its Chroma volume intact until the new version passes smoke tests.
3. Rollback steps:
   ```bash
   docker compose stop app
   # point compose at the previous tag (image: papeer:<prev-sha>) or:
   docker run --rm -p 8501:8501 --env-file .env \
     -v papeer-chroma:/app/data/chroma_db papeer:<prev-sha>
   curl -fsS http://localhost:8501/_stcore/health   # confirm "ok"
   ```
4. If the rollback was triggered by a **bad index** (not bad code): restore the last good
   backup (§5) or simply re-run ingestion from source documents.
5. Record what broke, the trigger, and the decision in the incident log before closing.
