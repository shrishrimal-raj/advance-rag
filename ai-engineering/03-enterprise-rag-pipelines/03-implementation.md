# 🔧 Week 3 Implementation — Build The Enterprise Search Engine

> Step-by-step. Each step ends with a **verify** line. Follow top to bottom.

## 0. Setup
```bash
cd ai-engineering
# deps already synced; local MiniLM cached
```
**verify:** `../.venv/bin/python -c "import rank_bm25, sentence_transformers; print('ok')"`

## 1. Build per-tenant BM25 index
Tokenize the tenant's passages, build `BM25Okapi`. Keep it in memory, keyed by tenant.
**verify:** `bm25.get_scores([some_query])` returns a float per passage.

## 2. Dense retrieval
Embed passages + query (local MiniLM), cosine top-k. Store vectors + metadata.
**verify:** dense top-1 for an exact-term query is *worse* than BM25 (shows why we need both).

## 3. Fuse with RRF
`rrf(list_a, list_b, k=60)` -> merged ranked ids. Implement from scratch (see approach_3).
**verify:** a doc ranked high in both lists outranks a doc high in only one.

## 4. Optional cross-encoder re-rank
`--rerank` loads `get_reranker()` (~80 MB download) and re-scores the fused top-N, keeping top-k. Off by default.
**verify:** with `--rerank`, the truly-relevant doc moves to #1 on a hard query.

## 5. The four approaches
| File | Proves |
|------|--------|
| `code/main.py` | Full hybrid + RRF (+opt-in rerank), multi-tenant, cited results |
| `code/approach_2_fastembed_hybrid.py` | Framework-free hybrid; `--backend st|fastembed` |
| `code/approach_3_rrf_from_scratch.py` | The RRF math, pure python |
| `code/benchmark_naive_vs_hybrid.py` | naive vs hybrid vs +rerank on a gold corpus |

## 6. Run everything
```bash
PY=../.venv/bin/python   # or .venv\Scripts\python.exe
$PY code/approach_3_rrf_from_scratch.py          # offline
$PY code/benchmark_naive_vs_hybrid.py            # offline (rerank opt-in)
$PY code/main.py --tenant acme --query "ERR-4042"
$PY code/approach_2_fastembed_hybrid.py          # offline (st backend default)
```

## Troubleshooting
- **Rerank download** — only happens with `--rerank`; skip it to stay offline.
- **BM25 empty corpus** — guard `if not passages: return []` before building the index.
- **Exact term not found** — confirm the token exists verbatim; BM25 needs the literal token.
- **fastembed backend** — first use downloads a model; default to `st` to stay offline.
