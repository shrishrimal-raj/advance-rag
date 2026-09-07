# 🔨 Module 1 Implementation — Build Your First RAG Pipeline

You will build a naive RAG pipeline step by step. The finished script is `code/main.py` — read it alongside these steps.

## Step 0 — Environment setup
```bash
uv sync                      # install all dependencies (first time only)
# optional, for local LLM:
ollama serve                 # terminal 1
ollama pull llama3.1         # terminal 2 (one-time download)
```
**Why:** `uv sync` creates `.venv` from `pyproject.toml` + `uv.lock`, giving a reproducible environment. Ollama gives us a free local LLM so no API key is needed.

## Step 1 — Load documents
```python
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
docs = TextLoader("data/samples/rag_overview.txt").load() + \
       UnstructuredMarkdownLoader("data/samples/vector_db_notes.md").load()
```
**Why:** Loaders normalize any file format into LangChain `Document` objects (`page_content` + `metadata`). Each document carries its source path in metadata — this is what powers citations later.

## Step 2 — Split into chunks
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```
**Why:** LLMs have limited context and embeddings work best on focused passages. ~300 chars ≈ 75 tokens is a good naive default. Overlap prevents cutting a sentence mid-thought. (Module 2 goes deep on this.)

## Step 3 — Embed and store in a vector database
```python
from langchain_chroma import Chroma
from shared.config import get_embeddings
vs = Chroma.from_documents(chunks, get_embeddings(), collection_name="module1")
```
**Why:** Embeddings turn text into 384-dim vectors where semantic similarity ≈ geometric closeness. ChromaDB stores them with an HNSW index for fast approximate nearest-neighbor search. Using the **same** `get_embeddings()` for storage and querying is critical.

## Step 4 — Retrieve
```python
retriever = vs.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What indexing strategy does HNSW use?")
```
**Why:** The retriever embeds the question and returns the 3 closest chunks. This is the "R" in RAG.

## Step 5 — Generate a grounded answer
```python
from shared.config import get_llm
prompt = f"""Use ONLY the context below to answer. Cite sources like [1], [2].
Context:
{formatted_context}
Question: {question}"""
answer = get_llm().invoke(prompt)
```
**Why:** "Use ONLY the context" is the anti-hallucination instruction. Numbered context blocks make citations possible and verifiable.

## Step 6 — Run it
```bash
uv run python 01-rag-fundamentals/code/main.py
```
Expected output: for each example question → the 3 retrieved chunks (with source + score) → the generated answer with `[n]` citations → finally a "why this is naive" summary.

## Experiments to try (do at least 2)
1. Set `chunk_size=50` → watch retrieval quality collapse (chunks too small to be self-contained)
2. Set `k=10` → more context, but notice irrelevant chunks diluting the answer
3. Ask a question about something NOT in the corpus (e.g., "Who won the 2022 World Cup?") → observe the model either refuse or hallucinate. This motivates threshold retrievers (Module 5)
4. Swap the question vocabulary: "How does ANN indexing work?" vs "Explain HNSW" → see how semantic search bridges vocabulary gaps

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `ConnectionError ... localhost:11434` | Start Ollama: `ollama serve`, then `ollama pull llama3.1` |
| First run is slow | Embedding model (~90 MB) downloads on first use — normal |
| `ModuleNotFoundError: shared` | You ran from inside `code/`; run from project root instead |
