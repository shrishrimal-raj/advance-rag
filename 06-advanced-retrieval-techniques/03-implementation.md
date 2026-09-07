# 🔨 Module 6 — Implementation Guide

You will build a **metadata-enriched corpus** from `data/samples/` and run **four** advanced
retrievers on it. Everything lives in `code/main.py`.

> Run from the project root:
> ```bash
> uv run python 06-advanced-retrieval-techniques/code/main.py
> ```
> Retriever #2 works with no LLM. Retriever #1, #3, #4 need a running LLM
> (`ollama serve` + `ollama pull llama3.1`, or set `OPENAI_API_KEY`).

---

## Step 1 — Bootstrap & imports

```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm, get_embeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import (
    ContextualCompressionRetriever, ParentDocumentRetriever,
    SelfQueryRetriever, MultiQueryRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore
from pydantic import BaseModel, Field
```

We now import **`get_llm`** (new this module) alongside `get_embeddings`.

## Step 2 — Build a metadata-enriched corpus

Self-Query needs something to filter on, so every chunk carries **three** metadata fields:

```python
TOPICS = {
    "rag_overview.txt":     "RAG fundamentals",
    "vector_db_notes.md":   "Vector databases & indexing",
    "company_profile.json": "Company profile & robotics products",
    "products.csv":         "Product catalog & pricing",
}
# per document: metadata = {"source": name, "format": ext, "topic": TOPICS[name]}
```

`build_main_vectorstore()` chunks (`chunk_size=300, overlap=40`), stamps a stable `id`, and embeds
into an in-memory Chroma collection (`m6_main`, cosine space). We keep the raw `docs` list too —
the Parent-Document retriever re-ingests it at its own chunk sizes.

## Step 3 — Define the Self-Query schema

A Pydantic model tells the LLM exactly which fields exist and what they mean:

```python
class DocSchema(BaseModel):
    source: str = Field(description="Source filename, e.g. 'products.csv'")
    format: str = Field(description="File format: txt, md, csv or json")
    topic:  str = Field(description="High-level topic of the document")
```

The descriptions matter — they're what the LLM reads when deciding how to filter.

## Step 4 — Retriever 1: Contextual Compression

```python
compressor = LLMChainExtractor(llm=get_llm())          # LLM keeps only relevant sentences
base       = vs.as_retriever(search_kwargs={"k": 4})  # retrieve MORE than needed
comp       = ContextualCompressionRetriever(base_retriever=base, base_compressor=compressor)
docs = comp.invoke(question)
```

Observe the returned text is **shorter** than the raw chunks — that's the token saving.

## Step 5 — Retriever 2: Parent-Document (small-to-big)

```python
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200,  chunk_overlap=0)
parent_store    = InMemoryStore()                      # holds the big parents, keyed by id
child_vs        = Chroma(embedding_function=get_embeddings(),
                         collection_name="m6_children",
                         collection_metadata={"hnsw:space": "cosine"})
retriever = ParentDocumentRetriever(
    vectorstore=child_vs, parent_doc_store=parent_store,
    child_splitter=child_splitter, parent_splitter=parent_splitter)
retriever.add_documents(docs)      # splits into parents+children, wires parent_id
parents = retriever.invoke(question)  # matches children, returns PARENTS
```

Note the **two stores**: children live in the vector store (for matching), parents live in the
`InMemoryStore` (for retrieval-by-id). `add_documents` builds both and links them via `parent_id`.

## Step 6 — Retriever 3: Self-Query

```python
sq = SelfQueryRetriever.from_llm(llm=get_llm(), vectorstore=vs, document_prompt=DocSchema)
docs = sq.invoke("Show me electronics products from the CSV catalog.")
```

The LLM parses the sentence into a query string **plus** a filter like `{"format": "csv"}`, and the
vector search runs with that filter applied. Watch the results come only from the right source.

## Step 7 — Retriever 4: Multi-Query

```python
base = vs.as_retriever(search_kwargs={"k": 2})
mq   = MultiQueryRetriever.from_llm(llm=get_llm(), retriever=base)  # default prompt -> 3 queries
docs = mq.invoke("Tell me about Acme Robotics and its products.")
```

The default prompt generates **3** diverse rewrites (our `num_queries=3`), retrieves per query, and
dedupes/merges. Bump the count by customizing the parser prompt if you want more recall.

## Step 8 — Graceful LLM failure

Every LLM-backed demo is wrapped in `try/except`. On failure we print a friendly panel:

```
⚠ Could not reach the LLM. Start Ollama locally:
    ollama serve
    ollama pull llama3.1
then re-run this module.
```

So a missing Ollama never crashes the script — the non-LLM parts still run and you see a clear hint.

---

## 🧪 Suggested experiments

1. **Parent/child sizes:** try `parent=2000, child=100` vs `parent=500, child=150`; watch match precision vs context length.
2. **Compression k:** raise the base `k` to 6 — does the extractor find more gold, or just more noise?
3. **Self-Query filters:** ask for "vector database notes that are markdown" and confirm `format=md` is applied.
4. **Multi-Query N:** increase rewrites to 5; measure latency and duplicate rate.

Record observations in `notes.md`.
