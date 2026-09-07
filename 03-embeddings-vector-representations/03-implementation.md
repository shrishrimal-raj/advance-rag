# 🔨 Module 3 — Implementation Guide

> Build a hands-on embeddings lab in `code/main.py`. No LLM, no API keys — just the shared local
> embedding model + NumPy. Follow the steps top-to-bottom; each maps to a block in the code.

**Run it:**
```bash
uv run python 03-embeddings-vector-representations/code/main.py
```

---

## Step 0 — Bootstrap the environment

Open `code/main.py`. The first lines make the project root importable so we can reuse the shared
config from anywhere:

```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_embeddings
```

- `parents[2]` walks up `code/ → module-folder/ → project-root/`.
- `get_embeddings()` returns the local `sentence-transformers/all-MiniLM-L6-v2` model (384-dim)
  unless `.env` sets `EMBEDDING_PROVIDER=openai`. First run downloads the model (~90 MB).

## Step 1 — Define sentences across 3 topics

Create a list of ~8 short sentences spanning **three distinct topics** (RAG/AI, cooking, sports)
plus a couple of cross-topic distractors. Keep parallel `labels` (`S1…S8`) for readable matrices.

> **Why 3 topics?** Clustering is only visible when there are clearly separate groups to separate.

## Step 2 — Embed them

```python
emb = get_embeddings()
vectors = np.array(emb.embed_documents(sentences), dtype=np.float32)
dim = vectors.shape[1]
```

- `embed_documents(list[str]) -> list[list[float]]`; wrap in `np.array` → shape `(N, dim)`.
- Wrap this in a `try/except` and print a helpful message if the model can't download/load.

## Step 3 — Compute the three pairwise matrices

Write three tiny metric functions, then fill an `N×N` matrix for each:

```python
def cosine_similarity(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
def l2_distance(a, b):       return float(np.linalg.norm(a - b))
def inner_product(a, b):     return float(np.dot(a, b))
```

Loop `i, j` over all pairs and store the result. Render each matrix as a `rich.Table` so the
diagonal and same-topic blocks are easy to read.

> **What to look for:** cosine diagonal = `1.0`; L2 diagonal = `0.0`; same-topic cells are the
> most similar (highest cosine / lowest L2).

## Step 4 — Prove cosine ≈ inner product after normalization

```python
def l2_normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)

normed = l2_normalize(vectors)
ip_normed = normed @ normed.T          # inner products on unit vectors
max_diff = float(np.max(np.abs(ip_normed - cos_m)))
```

Print `max_diff` — it should be ≈ `0` (floating-point noise). This is the numeric proof that
**normalizing makes the dot product equal cosine similarity** (see `02-learning.md` §4.5).

## Step 5 — Nearest-neighbour lookup for a query

```python
query = "How does RAG reduce hallucination in large language models?"
qv = np.array(emb.embed_query(query), dtype=np.float32)
sims = np.array([cosine_similarity(qv, v) for v in vectors])
order = np.argsort(-sims)             # descending
```

Print the top-3 neighbours with their scores. Add a soft sanity check: the #1 hit should be a
RAG/AI sentence. Use `embed_query` (not `embed_documents`) for a single query string.

## Step 6 — Report dimensionality + model note

Print `dim` and a short panel explaining the trade-off (384-dim MiniLM = fast/free but weaker than
1024/1536-dim models) and pointing to the comparison table in `02-learning.md` §6.

---

## ✅ Verification checklist

- [ ] Script runs with no errors and prints all 5 numbered steps.
- [ ] Three matrices render; diagonals are correct (cosine `1.0`, L2 `0.0`).
- [ ] `max |IP_normalized − cosine|` prints as ≈ `0`.
- [ ] Nearest-neighbour top hit is a RAG/AI sentence.
- [ ] Dimension prints as `384` (for the default MiniLM model).

## 🧪 Experiments to try

1. Replace a cooking sentence with a RAG sentence and watch the cosine matrix re-cluster.
2. Set `EMBEDDING_PROVIDER=openai` in `.env` (needs a key) and compare dimensions (1536) + scores.
3. Print `np.linalg.norm(vectors, axis=1)` to see whether the model normalizes by default.
4. Try a query that mixes two topics and observe how the scores split.
