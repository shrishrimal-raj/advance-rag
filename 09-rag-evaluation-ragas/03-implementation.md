# 🔨 Module 09 — Implementation: Evaluating a RAG Pipeline with RAGAS

This guide walks through the exact steps the two scripts perform. You can follow along by reading the code, or run each step and observe the output. Everything runs from the **project root** with `uv run python <path>` and works with **no API keys** (local Ollama + local embeddings).

**Files:**
- `code/build_dataset.py` → builds `data/eval_dataset.json` (golden set)
- `code/evaluate_pipeline.py` → scores the pipeline (individual + full RAGAS)

**Prereq:** `uv sync` done; Ollama running (`ollama serve`, `ollama pull llama3.1`) *or* `OPENAI_API_KEY` in `.env`.

---

## Step 0 — Project layout & conventions

Both scripts live in `09-rag-evaluation-ragas/code/` and start with the standard header so they can import the shared config from the project root:

```python
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))   # -> project root
from shared.config import get_llm, get_embeddings
```

Paths used (all resolved relative to the script, so it works from any CWD):
- `SAMPLES_DIR = <root>/data/samples/` (input documents)
- `DATASET_PATH = <module>/data/eval_dataset.json` (golden set, module-local)

---

## Step 1 — Build the corpus (load + chunk + embed)

The corpus is the four files in `data/samples/`. We load each as a `Document`, chunk it, embed it, and store it in an **in-memory Chroma** collection (no disk side-effects, so runs are reproducible and fast).

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:                      # older installs
    from langchain_community.vectorstores import Chroma

def load_documents(samples_dir):
    docs = []
    for f in sorted(samples_dir.iterdir()):
        if f.is_file():
            docs.append(Document(page_content=f.read_text(encoding="utf-8"),
                                 metadata={"source": f.name}))
    return docs

chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50) \
           .split_documents(load_documents(SAMPLES_DIR))
vectorstore = Chroma.from_documents(chunks, embedding_function=get_embeddings())
```

**Why in-memory?** Evaluation should be deterministic and not depend on a stale persisted DB. If you later evaluate a *production* index, point at that instead.

---

## Step 2 — Build the golden dataset

Run:
```bash
uv run python 09-rag-evaluation-ragas/code/build_dataset.py
```

What it does:
1. Reads the four sample files.
2. For **10 hand-written questions**, pulls the **actual source snippet** as `contexts` (e.g., the Falcon AMR JSON object, the specific CSV row, the "RAG solves two problems" paragraph, the HNSW section) — nothing fabricated.
3. Writes each example as `{question, ground_truth, contexts}` to `data/eval_dataset.json`.
4. Prints the dataset in a rich table so you can eyeball it.

Inspect the output and confirm each `ground_truth` is truly supported by its `contexts`. Tweak any that feel off — this is your reference signal, so make it right.

---

## Step 3 — Wire the retriever + generator (RAGAS-compatible callables)

RAGAS needs two things per example: the **retrieved contexts** and the **generated answer**. We expose them as clean callables that wrap our pipeline:

```python
TOP_K = 3
def retriever(question: str) -> list[str]:
    """Retrieval stage: question -> top-k context strings."""
    return [d.page_content for d in vectorstore.similarity_search(question, k=TOP_K)]

PROMPT = """Answer using ONLY the context below. If it isn't there, say you don't know. Be concise.
Context:
{context}

Question: {question}
Answer:"""

def generator(question: str, contexts: list[str]) -> str:
    """Generation stage: question + contexts -> answer string."""
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_template(PROMPT) | get_llm(temperature=0.0)
    return chain.invoke({"context": "\n\n".join(contexts), "question": question}).content
```

These are the "retriever/generator callables" — they're what we feed into RAGAS (by pre-computing each example's contexts + answer, which is how `ragas >= 0.2`'s `evaluate()` consumes data).

---

## Step 4 — Run individual metrics on 2 examples

For fast, targeted debugging, score **one metric at a time** on a couple of examples. This shows you exactly where a single answer stands.

```python
import asyncio
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

judge_llm  = LangchainLLMWrapper(get_llm())
embedder   = LangchainEmbeddingsWrapper(get_embeddings())

metrics = {
    "faithfulness":      Faithfulness(),
    "answer_relevancy":  AnswerRelevancy(),
    "context_precision": ContextPrecision(),
    "context_recall":    ContextRecall(),
}
for m in metrics.values():
    m.llm = judge_llm
metrics["answer_relevancy"].embeddings = embedder   # AR needs embeddings

def make_sample(ex):
    ctxs = retriever(ex["question"])
    return SingleTurnSample(
        user_input=ex["question"],
        retrieved_contexts=ctxs,
        reference=ex["ground_truth"],
        response=generator(ex["question"], ctxs),
    )

def score(metric, sample):
    # ragas >= 0.2 async single-sample API (falls back gracefully)
    if hasattr(metric, "single_turn_asynccreate"):
        return asyncio.run(metric.single_turn_asynccreate(sample, None))
    return metric.score(sample)

for ex in dataset[:2]:
    s = make_sample(ex)
    for name, m in metrics.items():
        print(f"{name}: {score(m, s):.3f}")
```

**Reading the numbers:** each is in `[0, 1]`. Faithfulness near 1.0 = grounded; Answer Relevancy high = on-topic; Context Precision high = good ranking; Context Recall high = complete retrieval. A low number here tells you *which* stage to fix (see the decision tree in `02-learning.md` §9).

> ⏱️ Each metric = LLM judge call(s). On local Ollama this is slow — expect seconds-to-tens-of-seconds per metric.

---

## Step 5 — Run the full `ragas.evaluate()` over the dataset

For the aggregate scorecard, bundle all metrics and run over every example:

```python
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset

ragas_ds = EvaluationDataset([make_sample(ex) for ex in dataset])
result = evaluate(
    dataset=ragas_ds,
    metrics=list(metrics.values()),
    llm=judge_llm,
    embeddings=embedder,
)
print(result)                 # per-example table
df = result.to_pandas()       # means / trends
```

`evaluate_pipeline.py` prints both a **per-example table** and a **mean-score summary** via `rich`.

> ⏱️ **This is the slow part.** 10 examples × 4 metrics × (1–3 judge calls each) on a local LLM can take many minutes. To learn faster, cap it: `EVAL_LIMIT=3 uv run python .../evaluate_pipeline.py`. For real speed, set `OPENAI_API_KEY` in `.env`.

---

## Step 6 — Interpret results & iterate

1. **Look at the mean table.** Which metric is lowest? That's your bottleneck stage.
2. **Map it to a fix** (decision tree, `02-learning.md` §9):
   - Low **Context Recall** → raise `TOP_K`, fix chunking, better embeddings, hybrid search.
   - Low **Context Precision** → add a reranker, lower `TOP_K`, metadata filters.
   - Low **Faithfulness** → tighten the prompt ("use ONLY context"), lower temperature, stronger model.
   - Low **Answer Relevancy** → sharper prompt, force direct/short answers.
3. **Change one thing**, re-run, and confirm the target metric moved while others didn't regress.
4. **Record the baseline** (means) in `notes.md` and set CI floors/regression margins (§8 of the learning doc).

### Quick "break it on purpose" experiment (proves the metric works)
- Set `TOP_K = 1` → watch **Context Recall** drop (incomplete retrieval).
- Remove "use ONLY the context" from the prompt → watch **Faithfulness** drop (hallucination).
- Swap the generator for a chatty model/high temperature → watch **Answer Relevancy** drop.

If a change you *know* is bad doesn't move the corresponding metric, your evaluator or golden set is mis-specified — investigate before trusting it.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ConnectionError` to `localhost:11434` | Ollama not running | `ollama serve`; `ollama pull llama3.1` |
| Import error on `ragas.metrics` | RAGAS version mismatch | Check `ragas.__version__`; adjust imports (see note in code) |
| Very slow evals | Local LLM judge | Use OpenAI judge, or `EVAL_LIMIT` to cap examples |
| Scores jitter between runs | Judge non-determinism | Average runs; use thresholds with margin |
| `data/eval_dataset.json` missing | Didn't run builder first | Run `build_dataset.py` first |
