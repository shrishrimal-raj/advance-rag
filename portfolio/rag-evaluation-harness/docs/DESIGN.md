# DESIGN — RAG Evaluation & MLOps Harness

This document describes the architecture **as implemented** in this repository.
The harness is a batch CLI (not a service): it reads a golden dataset of
pre-retrieved `(question, contexts, answer)` rows, scores each row with
deterministic offline metrics (plus optional LLM-judged RAGAS metrics), and can
gate CI on metric regressions.

## 1. System Context

The harness sits between the RAG pipeline under test (upstream) and CI /
humans (downstream). It has no runtime dependencies on a vector store or an LLM
server; cloud LLMs are optional and only used for pipeline B.

```mermaid
flowchart TD
    subgraph Upstream["Upstream (not in this repo)"]
        PIPE[RAG pipeline under test<br/>retrieval + generation]
    end

    subgraph Harness["rag-evaluation-harness"]
        CLI[cli.py<br/>run / gate commands]
        ENV[.env<br/>optional API keys]
    end

    subgraph Downstream["Downstream consumers"]
        CI[CI pipeline<br/>blocks on gate exit 1]
        DEV[Developer<br/>reads report + JSON]
    end

    CLOUD[(Cloud LLM<br/>OpenAI or Yolo-Auto<br/>optional)]

    PIPE -->|emits rows: question,<br/>contexts, answer| DS[evals/dataset.jsonl]
    DS --> CLI
    ENV -.->|keys if present| CLI
    CLI -->|pipeline B only| CLOUD
    CLI -->|out/report.md<br/>out/results_A.json| DEV
    CLI -->|exit 0 / exit 1| CI
```

## 2. Component Diagram

Two modules plus the CLI entrypoint. `harness/config.py` owns all external
integration (LLM factories, embedder factory); `harness/metrics.py` owns pure
scoring logic and depends on `config` only for `cosine`.

```mermaid
flowchart LR
    subgraph cli["cli.py (entrypoint)"]
        RUN[cmd_run<br/>run_pipeline, write_report, print_table]
        GATE[cmd_gate<br/>baseline compare]
        LOAD[load_dataset<br/>JSONL parser]
    end

    subgraph cfg["harness/config.py"]
        GETLLM[get_llm / get_llm_provider_name<br/>OpenAI → Yolo-Auto → None]
        GETEMB[get_embedder<br/>MiniLM → HashingEmbedder]
        ST[SentenceTransformerEmbedder]
        HASH[HashingEmbedder<br/>pure-Python BoW hashing]
        COS[cosine]
    end

    subgraph met["harness/metrics.py"]
        TOK[tokenize / key_terms / split_sentences]
        M1[context_keyword_coverage]
        M2[answer_relevancy]
        M3[faithfulness_proxy]
        EQ[evaluate_question]
        RAGAS[ragas_scores<br/>graceful-skip wrapper]
    end

    OUT[out/report.md<br/>out/results_*.json]
    BASE[baseline JSON]

    RUN --> LOAD
    RUN --> GETEMB
    RUN -->|pipeline B only| GETLLM
    RUN --> EQ
    RUN --> RAGAS
    GATE --> RUN
    GATE --> BASE
    EQ --> M1 & M2 & M3
    M1 --> TOK
    M2 --> COS
    M3 --> COS
    GETEMB --> ST
    GETEMB -->|fallback| HASH
    EQ --> OUT
```

## 3. Retrieval / Data-Flow Sequence (one dataset row, pipeline A)

Retrieval itself happens upstream; the sequence below shows what the harness
does per row: embed once per text, score three metrics, accumulate.

```mermaid
sequenceDiagram
    participant C as cli.run_pipeline
    participant E as get_embedder()
    participant M as evaluate_question()
    participant K as context_keyword_coverage
    participant A as answer_relevancy
    participant F as faithfulness_proxy

    C->>E: prefer_local_model=True
    alt MiniLM cached
        E-->>C: SentenceTransformerEmbedder
    else not cached / offline
        E-->>C: HashingEmbedder (deterministic fallback)
    end
    loop each row in dataset.jsonl
        C->>M: question, contexts, answer, embed
        M->>K: key_terms(question) vs joined contexts
        K-->>M: hits/len(terms), clamped [0,1]
        M->>A: cosine(embed(question), embed(answer))
        A-->>M: similarity, clamped [0,1]
        M->>F: per-sentence max cosine vs context vectors
        F-->>M: mean claim-context similarity
        M-->>C: {context_keyword_coverage, answer_relevancy, faithfulness_proxy}
    end
    C->>C: averages = mean per metric (rounded to 4 dp)
    C-->>C: Rich table + out/report.md + out/results_A.json
```

Pipeline B inserts one extra step per row after `evaluate_question`:
`ragas_scores(question, reference_answer, contexts, answer, llm)` →
`{ragas_faithfulness, ragas_answer_relevancy}` or `{}` when unavailable, plus a
`ragas_available` flag on the row.

## 4. Deployment Topology

The harness deploys as a **batch job**, not a long-running service. There are
no exposed ports. In Docker it runs once, writes artifacts to a mounted volume,
and exits; in GitHub Actions it runs directly via `uv`.

```mermaid
flowchart TB
    subgraph GH["GitHub Actions runner (ubuntu)"]
        CO[checkout] --> SU[setup-python 3.11 + uv]
        SU --> SYNC[uv sync]
        SYNC --> PYT[uv run pytest -v]
        PYT -->|non-zero exit| FAIL[[build fails]]
        PYT -->|pass| DONE[[build succeeds]]
    end

    subgraph DOCKER["Docker (local / self-hosted)"]
        IMG[Dockerfile: python:3.11-slim + uv<br/>uv sync --frozen]
        COMPOSE[docker-compose.yml<br/>env_file: .env]
        VOL[(./out volume<br/>report.md + results_A.json)]
        COMPOSE -->|docker compose run harness| JOB[harness container<br/>cli.py run --pipeline A]
        JOB --> VOL
        IMG -.build.-> JOB
    end

    KEY[(.env — never committed,<br/>never baked into image)] -.->|mounted at runtime| COMPOSE
```

Ports: **none**. The container exposes nothing; results are consumed from the
`./out` bind mount (or `docker compose logs`).

## 5. Failure Modes & State Diagram

Every external dependency has a defined degradation path so that pipeline A
never crashes. The state diagram tracks one evaluation run through its
possible states.

```mermaid
stateDiagram-v2
    [*] --> LoadingDataset
    LoadingDataset --> Scoring: rows parsed
    LoadingDataset --> Failed: file missing / bad JSONL
    Scoring --> SelectingEmbedder
    SelectingEmbedder --> ScoringRows: MiniLM cached
    SelectingEmbedder --> ScoringRows: HashingEmbedder fallback
    ScoringRows --> RagasStep: pipeline B
    ScoringRows --> WritingArtifacts: pipeline A
    RagasStep --> WritingArtifacts: ragas ok or skipped ({})
    WritingArtifacts --> GateCheck: gate command
    WritingArtifacts --> Done: run command
    GateCheck --> Done: no drop > tolerance (exit 0)
    GateCheck --> Failed: drop > tolerance (exit 1)
    Failed --> [*]
    Done --> [*]
```

Failure-mode table:

| Failure | Detection | Behavior |
|---------|-----------|----------|
| No `.env` / no keys | `os.getenv` empty | `get_llm()` → `None`; pipeline B degrades to A's metrics with `ragas_available=false` |
| MiniLM not cached / download blocked | `SentenceTransformer(...)` raises | `get_embedder` catches → `HashingEmbedder` |
| `ragas`/`datasets` import fails | ImportError in wrapper | `ragas_scores` returns `{}` |
| RAGAS call fails at runtime (rate limit, schema drift) | Exception in wrapper | `ragas_scores` returns `{}` — run continues |
| Empty dataset | `load_dataset` returns `[]` | `cmd_run` prints error, returns 1 |
| Baseline missing a metric key | `k in base_avg` check | That metric is simply not gated |
| Windows console encoding | `sys.platform == "win32"` | stdout/stderr reconfigured to UTF-8 up front |
| Any unhandled exception in a command | top-level try/except in `main()` | Printed via Rich, exit 1 |

## 6. Key Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Batch CLI instead of a service | The job is "score a dataset and exit"; a server would add ports, lifecycle, and health concerns with zero benefit |
| D2 | Custom deterministic metrics as the primary gate signal | Reproducible across machines/runs → safe as a CI gate; LLM judges are too noisy/costly for every commit |
| D3 | Two-tier LLM selection (OpenAI → Yolo-Auto → None) | One code path (`ChatOpenAI`) serves both providers since Yolo-Auto is OpenAI-compatible; `None` keeps everything runnable keyless |
| D4 | Ollama deliberately excluded | Target machine is a low-power laptop without a local LLM server; a heuristic/offline mode covers the no-key case |
| D5 | `HashingEmbedder` as guaranteed fallback | Pure Python, zero deps, deterministic — the harness must *always* run, even with no model cache and no network |
| D6 | RAGAS wrapped in a never-raises adapter | Optional dependency must not be able to break the core path; contract is "dict or empty dict" |
| D7 | Baseline = plain JSON committed beside the dataset | Versioned with the code, diffable in review, regenerable from any known-good `results_A.json` |
| D8 | Metrics clamped to [0, 1] and rounded to 4 dp | Uniform scale makes tolerance-based gating meaningful; rounding stabilizes JSON diffs |

## 7. Trade-off Tables

### Embedding strategy

| Option | Quality | Offline guarantee | Speed | Chosen? |
|--------|---------|-------------------|-------|---------|
| Cached MiniLM (`all-MiniLM-L6-v2`) | Good semantic similarity | Only when cached (~90 MB) | Fast (CPU) | Yes — preferred |
| `HashingEmbedder` (BoW hashing, dim 256) | Weak (lexical overlap only) | Always | Instant | Yes — fallback |
| Cloud embedding API | Best | No (needs key + network) | Network-bound | No — violates offline-first goal |

### Metric tiering

| Tier | Cost per run | Determinism | What it catches | Used by |
|------|--------------|-------------|-----------------|---------|
| Custom offline (3 metrics) | ~0 | Exact | Lexical coverage, gross relevance/grounding drops | Pipeline A, CI gate |
| RAGAS (LLM-judged) | API cost | Stochastic | Subtle faithfulness/relevancy issues | Pipeline B, manual review |

### Packaging

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| GitHub Actions + `uv` | Free, reproducible lockfile, no image build | Runner-only | Yes (ci.yml) |
| Docker one-shot job | Portable to any host, pinned base image | Image build time; artifacts need a volume | Yes (Dockerfile + compose) |
| Long-running containerized service | Persistent logs/UI | No consumer exists; over-engineered | No |
