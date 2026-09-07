# 🔨 Module 2 Implementation — Loaders, Splitters & Metadata Lab

The finished script is `code/main.py`. It does three things: (1) ingests every sample file with the right loader, (2) compares 4 splitter strategies on the same document, (3) demonstrates metadata filtering.

## Step 0 — Environment
```bash
uv sync
```
No LLM needed for this module — it's pure text processing. Runs instantly.

## Step 1 — Format-aware loading
```python
loaders = {
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".json": JSONLoader,
    ".md": UnstructuredMarkdownLoader,
}
```
**Why:** each format has structural semantics (rows, keys, headers). Using the matching loader preserves them instead of flattening everything to raw text. Note the script attaches `format` and `source` metadata to every document immediately after loading.

## Step 2 — Side-by-side splitter comparison
The script takes `vector_db_notes.md` and runs it through:
1. `CharacterTextSplitter(chunk_size=300)` — hard cuts
2. `RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)` — separator-aware
3. `MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2")])` — structure-preserving
4. A custom section splitter (split on blank lines) — shows how to write your own

For each it prints: chunk count, min/max/avg size, and the first chunk's preview.
**What to observe:** the character splitter will cut mid-sentence; the markdown splitter keeps `##` sections together and stores the header path in metadata.

## Step 3 — Recommendation logic
The script ends with a printed recommendation table mapping document type → recommended splitter. This mirrors real ingestion-service config.

## Step 4 — Metadata filtering demo
```python
pdf_like = [c for c in chunks if c.metadata.get("format") == "csv"]
```
**Why this matters now:** in Modules 5–6, retrievers accept `filters={"format": "csv"}` — Self-Query Retriever even generates these filters from natural language. Filtering only works if metadata was attached here, correctly and consistently.

## Run it
```bash
uv run python 02-document-processing-chunking/code/main.py
```

## Experiments (do at least 2)
1. Drop `chunk_overlap` to 0 in the recursive splitter → find a sentence that gets split across two chunks
2. Change `chunk_size` from 300 → 100 and 1000 → compare avg chunk size and whether previews look self-contained
3. Add a new file `data/samples/my_doc.txt` → confirm it appears in the ingestion report
4. Try `MarkdownHeaderTextSplitter` on the plain-text file → see why format-aware selection matters

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `unstructured` import errors on some platforms | It's a heavy dep; if install fails, `uv sync` again or remove it from pyproject and skip the MD loader |
| CSV rows come out as separate documents | That's intended — one row = one searchable record |
| JSON produces one giant document | Normal for small files; for large nested JSON use `JSONLoader` with `jq` selectors |
