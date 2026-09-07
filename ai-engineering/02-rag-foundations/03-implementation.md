# 🔧 Week 2 Implementation — Build The Knowledge Engine

> Step-by-step. Each step ends with a **verify** line you can actually run. Follow top to bottom.

## 0. Setup
```bash
cd ai-engineering
uv sync                      # installs deps into ../.venv (already done in this repo)
cp .env.example .env         # ensure YOLO_AUTO_API_KEY is present (root .env also works)
```
**verify:** `../.venv/bin/python -c "import langchain_chroma, sentence_transformers; print('ok')"`

## 1. Load & chunk documents
Use `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)`. Keep each chunk a coherent passage; tag with metadata `{tenant, source}`.
**verify:** print `len(chunks)` — should be > number of source docs (splitting happened).

## 2. Embed + store in a per-tenant Chroma collection
`get_embeddings()` returns local MiniLM (cached). Store into collection `knowledge_<tenant>` under `CHROMA_DIR`. Delete the collection first so re-runs don't duplicate.
**verify:** `collection.count()` equals number of chunks.

## 3. Retrieve top-k with tenant filter
`similarity_search_with_relevance_scores(question, k=3)`. Because the collection is per-tenant, isolation is structural.
**verify:** the top hit for "refund policy" is the refund passage.

## 4. Generate a grounded answer
Prompt template: "Answer using ONLY the context… cite the source." Call `get_llm().invoke(...)`. Read `usage_metadata` for token accounting.
**verify:** `python code/main.py --selftest` prints retrieved context, an answer, and token count.

## 5. The four approaches
| File | What it proves |
|------|----------------|
| `code/main.py` | Full LangChain RAG, per-tenant, grounded answer + cost |
| `code/approach_2_raw_vector_db.py` | Same retrieval with the raw Chroma client (no LangChain) |
| `code/approach_3_cosine_from_scratch.py` | The math: TF-IDF + cosine top-k in numpy, no model/API |
| `code/chunking_benchmark.py` | Which chunking strategy retrieves the gold passage best |

## 6. Run everything
```bash
PY=../.venv/bin/python   # or .venv\Scripts\python.exe on Windows
$PY code/approach_3_cosine_from_scratch.py     # offline, prints top-k + self-check
$PY code/chunking_benchmark.py                 # offline, prints strategy table
$PY code/approach_2_raw_vector_db.py           # cached MiniLM, no LLM
$PY code/main.py --selftest                    # one cloud LLM call
```

## Troubleshooting
- **Embedding download on first run** — MiniLM (~90 MB) is cached from prior work; if truly missing it downloads once. Never pull Ollama.
- **LLM unavailable** — `main.py` catches the error, prints a hint, exits 0 (graceful degradation).
- **Wrong retrieval** — shrink `chunk_size` or increase `k`; check the benchmark to see which strategy wins.
- **Duplicate results on re-run** — confirm the collection is deleted before `add_texts`.
