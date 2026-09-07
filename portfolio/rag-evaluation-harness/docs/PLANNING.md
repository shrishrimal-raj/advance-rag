# PLANNING — RAG Evaluation & MLOps Harness

## Problem Statement

RAG pipelines are hard to change safely: a tweak to chunking, retrieval, or the
prompt can silently degrade answer quality, and most teams only notice in
production. LLM-judged evaluators (e.g. RAGAS) catch subtle quality issues but
require API keys, cost money per run, and are non-deterministic — so they cannot
be the sole gate in CI. This project provides a **standalone, offline-first
evaluation harness** that scores a fixed golden dataset with deterministic custom
metrics, optionally layers on LLM-judged RAGAS scores, and exposes a
**regression gate** (`cli.py gate`) whose exit code can block a CI pipeline when
any metric drops beyond a tolerance.

## Goals

- G1. Score a versioned golden dataset (`evals/dataset.jsonl`, 8 rows of
  question / contexts / answer / reference_answer) with metrics that need **no
  API keys and no network** (`context_keyword_coverage`, `answer_relevancy`,
  `faithfulness_proxy`).
- G2. Offer opt-in LLM-judged RAGAS metrics (`--pipeline B`) that degrade
  gracefully to `{}` when no key or the `ragas` package is unavailable.
- G3. Provide a regression gate: compare current averages against a baseline
  JSON; exit 1 if any tracked metric drops more than `--tolerance` (default
  0.05), exit 0 otherwise.
- G4. Produce human-readable artifacts: a Rich console table, a Markdown report
  (`out/report.md`), and machine-readable results (`out/results_A.json`).
- G5. Run fully offline on a low-power laptop: local-first embeddings
  (cached MiniLM) with a pure-Python `HashingEmbedder` fallback, and no local
  LLM server dependency (Ollama intentionally excluded).
- G6. Keep the whole thing testable offline: `uv run pytest -v` passes with no
  network and no model downloads.

## Non-Goals

- NG1. Not a serving stack — there is no HTTP server, no FastAPI, no vector DB
  deployment. Chromadb is a declared dependency inherited from the course
  template but the harness itself evaluates pre-retrieved `(question, contexts,
  answer)` rows.
- NG2. Not a retrieval implementation — no BM25/hybrid/RRF/rerank code lives in
  this repo; those are evaluated *upstream* by feeding their outputs into the
  dataset.
- NG3. Not a general experiment tracker — baselines are plain JSON files
  committed next to the dataset, not a database.
- NG4. No live-LLM calls in tests; RAGAS paths are covered only via the
  graceful-degradation contract (`ragas_scores(..., llm=None) == {}`).

## Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| R1 | `cli.py run --pipeline A` scores every dataset row with the 3 offline metrics and writes table + report + JSON | Must | Done |
| R2 | `cli.py run --pipeline B` adds `ragas_*` columns when an LLM key is configured; sets `ragas_available` per row | Should | Done |
| R3 | `cli.py gate --baseline <json> --tolerance <t>` exits 1 on any drop > t, 0 otherwise | Must | Done |
| R4 | All custom metrics return values clamped to [0, 1] and are deterministic for identical inputs | Must | Done |
| R5 | Embedding selection: cached `all-MiniLM-L6-v2` first, `HashingEmbedder` fallback so runs never fail offline | Must | Done |
| R6 | LLM selection: OpenAI → Yolo-Auto (OpenAI-compatible) → `None` (heuristic/offline mode) | Must | Done |
| R7 | `ragas_scores` never raises; any import/runtime failure returns `{}` | Must | Done |
| R8 | Test suite passes offline with zero model downloads | Must | Done |
| R9 | Golden dataset stays small (8–10 rows) and is versioned with the code | Should | Done (8 rows) |
| R10 | Docker + GitHub Actions packaging so the gate runs in CI | Should | In progress |

## Milestones

| Milestone | Scope | Exit criteria |
|-----------|-------|---------------|
| M1 — Core metrics | `harness/metrics.py`: tokenize/key_terms, 3 offline metrics, `evaluate_question` | Unit tests green offline |
| M2 — Config & fallbacks | `harness/config.py`: 2-tier LLM selection, embedder factory with hashing fallback | Runs with no `.env` at all |
| M3 — CLI & artifacts | `cli.py run`: Rich table, `out/report.md`, `out/results_*.json` | Report matches per-question scores |
| M4 — Regression gate | `cli.py gate` with baseline JSON + tolerance | Exit code 1 on injected regression |
| M5 — Optional RAGAS | `ragas_scores` wrapper with graceful skip | `llm=None` returns `{}`; no crash without key |
| M6 — Packaging & CI | Dockerfile, docker-compose.yml, `.github/workflows/ci.yml`, docs | `uv run pytest -v` green in Actions |

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MiniLM model not cached → first `run` downloads ~90 MB | Medium | Low | `get_embedder` falls back to `HashingEmbedder`; tests use the hashing embedder directly, so CI never downloads |
| RAGAS/langchain API drift breaks pipeline B | Medium | Low | All RAGAS imports/calls wrapped in try/except → `{}`; pipeline A unaffected |
| Hashing embedder is weak semantically → `answer_relevancy` noisy | High | Medium | Treated as a *gate* signal (relative drops), not an absolute quality score; MiniLM used when cached |
| Baseline drift: stale `baseline.json` masks regressions | Medium | High | Baseline regenerated deliberately from a known-good `results_A.json` `averages` block and committed with a note |
| Windows console encoding crashes on Unicode (Rich box-drawing) | Low | Low | `cli.py` reconfigures stdout/stderr to UTF-8 on `win32` |
| Live API key leaked into repo/artifacts | Low | High | `.env` git-ignored; `.env.example` holds placeholders only; tests make no network calls |

## Success Metrics

- **Gate reliability:** a deliberate quality drop > tolerance always yields exit
  1; a clean re-run of the same dataset yields exit 0 (determinism check).
- **Offline guarantee:** `uv run pytest -v` completes with network disabled and
  no model cache present.
- **Reproducibility:** two consecutive `run --pipeline A` invocations produce
  identical `averages` in `results_A.json`.
- **Coverage:** every public function in `harness/metrics.py` exercised by at
  least one test, including the RAGAS degradation path.
- **CI time:** full `uv sync && pytest` under ~5 minutes on a standard runner.
