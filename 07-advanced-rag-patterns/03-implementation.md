# Module 07: Advanced RAG Patterns — Implementation Guide

This guide walks through each subfolder's code step by step. All scripts are run from the **project root**:

```bash
uv run python 07-advanced-rag-patterns/code/<pattern>/main.py
```

> **Prerequisite:** Make sure Ollama is running (`ollama serve`) with `llama3.1` pulled, or set `OPENAI_API_KEY` in `.env`. Embeddings use local `sentence-transformers/all-MiniLM-L6-v2` (no API key needed).

---

## 1. RAG Fusion — `code/rag_fusion/main.py`

### What it does (stage by stage)

| Stage | What happens | Key code |
|-------|-------------|----------|
| 1 | Load sample text files from `data/samples/`, split into chunks | `load_corpus()` |
| 2 | Build in-memory Chroma vector store | `Chroma(embedding_function=embeddings)` |
| 3 | LLM generates 3 query variants from the user question | `generate_query_variants()` |
| 4 | Retrieve top-5 for EACH variant independently | Loop calling `vector_store.similarity_search()` |
| 5 | Reciprocal Rank Fusion (k=60) merges the 3 ranked lists | `reciprocal_rank_fusion()` |
| 6 | Print fused ranking table with per-query rank columns | `rich.table.Table` |
| 7 | Generate final answer from fused top-3 context | `generate_answer()` |

### Key concepts to look for in the code

- **RRF formula**: `score(doc) = sum(1 / (60 + rank))` across all lists. A doc at rank 1 in two lists scores higher than a doc at rank 1 in one list.
- **Query variant prompt**: The LLM is asked to rephrase the question from different angles (synonyms, broader, narrower).
- **Deduplication**: If the same chunk appears in multiple retrieval results, RRF naturally boosts it.

### Try this experiment

Change `k=60` to `k=10` and observe how the ranking changes. Smaller k gives more weight to top ranks.

---

## 2. HyDE — `code/hyde/main.py`

### What it does (stage by stage)

| Stage | What happens | Key code |
|-------|-------------|----------|
| 1 | Load corpus + build Chroma store | Same pattern as RAG Fusion |
| 2 | LLM writes a hypothetical answer document | `generate_hypothetical_doc()` |
| 3 | Embed the hypothetical doc (NOT the user query) | `embeddings.embed_query(hypothetical)` |
| 4 | Retrieve using hypothetical doc as query vector | `vector_store.similarity_search_with_relevance_scores()` |
| 5 | Print explanation of WHY this helps vocabulary mismatch | Console output |
| 6 | Generate real grounded answer from retrieved chunks | `generate_answer()` |

### Key concepts to look for in the code

- The user's raw question is NEVER used as the search vector. Only the hypothetical doc's embedding is.
- The hypothetical doc prompt instructs the LLM to write "as if you are a document in the corpus" — this biases it toward corpus vocabulary.
- Compare: if you retrieved using the raw query instead, would you get different results? (The code prints both for comparison.)

### Try this experiment

Remove the HyDE step and retrieve directly with the user query. Compare the top-3 results. You should see that HyDE retrieves more relevant chunks when vocabulary differs.

---

## 3. CRAG — `code/crag/main.py`

### What it does (stage by stage)

| Stage | What happens | Key code |
|-------|-------------|----------|
| 1 | Load corpus + build Chroma store | Standard setup |
| 2 | Retrieve top-3 documents | `vector_store.similarity_search(query, k=3)` |
| 3 | LLM validates each doc: Correct / Incorrect / Ambiguous | `validate_document()` per doc |
| 4 | Print validation decisions in a table | `rich.table.Table` |
| 5a | If enough Correct → generate answer from docs | `generate_answer_from_docs()` |
| 5b | If mostly bad → web fallback via DuckDuckGo HTML | `web_fallback()` using `requests` + `BeautifulSoup` |
| 5c | If web fails → "insufficient knowledge" response | Graceful degradation |
| 6 | Print final answer with source attribution | Console output |

### Key concepts to look for in the code

- **Validator prompt**: Asks the LLM to be strict. "Correct" means the doc directly answers the question, not just mentions related topics.
- **Web fallback**: Uses `https://html.duckduckgo.com/html/?q=...` (no API key needed). Parses `<h2 class="result__title">` and `.result__snippet` with BeautifulSoup.
- **Error handling**: The entire web call is in try/except. On any failure (network, parsing), it degrades gracefully.
- **Decision threshold**: At least 2 out of 3 docs must be "Correct" to skip the web fallback.

### Try this experiment

Ask a question that is NOT in the corpus (e.g., "What is the capital of France?"). Observe the validator marking docs as Incorrect and the web fallback kicking in.

---

## 4. Self-RAG — `code/self_rag/main.py`

### What it does (stage by stage)

| Stage | What happens | Key code |
|-------|-------------|----------|
| 1 | Load corpus + build Chroma store | Standard setup |
| 2 | Retrieve relevant chunks | `vector_store.similarity_search(query, k=5)` |
| 3 | Generate initial answer WITH citations | `generate_answer_with_citations()` |
| 4 | REFLECTION: LLM critiques answer vs evidence | `reflect_on_answer()` |
| 5 | Print reflection decisions (supported? relevant? complete?) | Console output |
| 6a | If critique passes → output initial answer | Done |
| 6b | If critique fails → regenerate ONCE with critique fed back | `regenerate_with_critique()` |
| 7 | Print final answer + reflection log | Console output |

### Key concepts to look for in the code

- **Citation format**: The initial answer must cite chunks like `[Chunk 1]`, `[Chunk 3]`. The reflection checks whether those citations actually support the claims.
- **Reflection prompt**: Asks three specific questions: (1) Is each claim supported by the cited evidence? (2) Is the answer relevant to the question? (3) Is it complete?
- **One regeneration only**: To avoid infinite loops, we regenerate at most once. The critique is appended to the generation prompt as "Previous attempt had these issues: ...".
- **Reflection output format**: Structured as PASS/FAIL per criterion, making it easy to parse and display.

### Try this experiment

Ask a question where the corpus has partial information. Observe whether the reflection catches the incompleteness and the regeneration adds a caveat.

---

## 5. GraphRAG — `code/graph_rag/main.py`

### What it does (stage by stage)

| Stage | What happens | Key code |
|-------|-------------|----------|
| 1 | Load corpus + chunk | Standard setup |
| 2 | For each chunk, LLM extracts entities + relations | `extract_entities_and_relations()` |
| 3 | Merge all extractions into a dict-of-dicts graph | `build_graph()` |
| 4 | Print graph nodes and edges | Console output |
| 5 | Find connected components via BFS (pure Python) | `find_communities_bfs()` |
| 6 | Print communities with their member entities | Console output |
| 7 | LLM summarizes each community | `summarize_community()` |
| 8 | Answer GLOBAL question using community summaries | `answer_global_question()` |
| 9 | Print final answer | Console output |

### Key concepts to look for in the code

- **Graph representation**: `graph = {entity: {related_entity: relation_string}}`. Undirected: if A→B exists, B→A also exists.
- **BFS for connected components**: Standard queue-based BFS. No networkx needed. Each unvisited node starts a new component.
- **Extraction prompt**: Asks for JSON output: `{"entities": [...], "relations": [{"from": ..., "to": ..., "relation": ...}]}`. Parsed with `json.loads()`.
- **Global question**: "What are the main themes across all documents?" — answered from community summaries, NOT raw chunks. This is the key difference from basic RAG.
- **LLM call count**: N_chunks (extraction) + N_communities (summaries) + 1 (final answer). For 4 chunks and 2 communities, that's 7 calls.

### Try this experiment

Add a 5th document to the corpus that introduces a new topic. Observe how a new community forms and how the global answer changes.

---

## 6. Multi-Modal RAG — `code/multimodal_rag/main.py`

### What it does (stage by stage)

| Stage | What happens | Key code |
|-------|-------------|----------|
| 1 | Create 3 labeled images with PIL (ImageDraw) | `create_demo_images()` |
| 2 | Try to load CLIP model (`clip-ViT-B-32`) | try/except around `SentenceTransformer(...)` |
| 3a | SUCCESS: Embed images + text into shared space | `model.encode(images)` / `model.encode(texts)` |
| 3b | FAILURE: Fall back to text-only demo explaining architecture | `text_only_fallback()` |
| 4 | Store embeddings in Chroma (images + text chunks) | `Chroma.add_embeddings()` |
| 5 | Cross-modal query: text → retrieve matching image | `vector_store.query(query_texts=[...])` |
| 6 | Cross-modal query: image → retrieve matching text | Encode image, use as query vector |
| 7 | Print results showing cross-modal matches | Console output |

### Key concepts to look for in the code

- **PIL image creation**: `Image.new('RGB', (200, 100), color)` + `ImageDraw.text()` to draw labels on colored boxes.
- **CLIP cross-modal space**: Both images and text are mapped to the same 512-dim vector. Cosine similarity between a text embedding and an image embedding is meaningful.
- **Graceful fallback**: ANY exception (download error, missing PIL, OOM) triggers the text-only demo. The script never crashes.
- **Production note** (in comments): Real systems use CLIP ViT-L/14 (~1.7 GB) or SigLIP for better alignment. ViT-B-32 is for quick demos.
- **DO NOT RUN during verification**: This script may download ~600 MB. Use `py_compile` only.

### Try this experiment (if you have GPU + disk space)

Run the script. Then change the text query from "a diagram about vector databases" to "a graph data structure" and see if a different image is retrieved.

---

## Common Patterns Across All Scripts

1. **Path setup**: Every script starts with `sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))` to import from project root.
2. **Windows encoding**: `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` for cp1252 consoles.
3. **LLM wrapper**: All LLM calls are in try/except with a friendly "Please start Ollama" message.
4. **Rich output**: `rich.console.Console`, `rich.panel.Panel`, `rich.table.Table` for pretty terminal output.
5. **In-memory Chroma**: `Chroma(embedding_function=embeddings)` with no persistent directory — fresh store every run.
6. **Corpus loading**: Reads from `data/samples/` (txt, md files) and splits into ~500-char chunks with overlap.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Ollama not reachable" | Run `ollama serve` in another terminal |
| "Model not found: llama3.1" | Run `ollama pull llama3.1` |
| Slow first run | Embedding model downloads on first use (~90 MB) |
| CRAG web fallback fails | Check internet; DuckDuckGo may rate-limit. Script degrades gracefully. |
| GraphRAG JSON parse error | LLM sometimes outputs non-JSON. Code has a fallback regex extractor. |
| Multi-Modal download stuck | Kill the process. The fallback will handle it on next run. Or pre-download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/clip-ViT-B-32')"` |
