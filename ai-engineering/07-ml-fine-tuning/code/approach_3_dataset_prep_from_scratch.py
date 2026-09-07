import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 3 - dataset preparation from scratch (offline).

Turns raw {prompt, completion} examples into a training-ready dataset: applies a
chat template, reports length stats, and does a stratified train/val split with no
leakage. Pure Python - no model needed.
"""

RAW = [
    {"prompt": "What is RAG?", "completion": "Retrieval-Augmented Generation grounds answers in retrieved documents.", "label": "rag"},
    {"prompt": "Explain embeddings.", "completion": "Embeddings map text to vectors so similar meanings sit close together.", "label": "embed"},
    {"prompt": "What is a vector store?", "completion": "A vector store indexes embeddings for fast similarity search.", "label": "rag"},
    {"prompt": "What is an embedding?", "completion": "An embedding is a dense numeric representation of a piece of text.", "label": "embed"},
    {"prompt": "How does retrieval work?", "completion": "Retrieval finds the most similar stored chunks to a query.", "label": "rag"},
    {"prompt": "Define a chunk.", "completion": "A chunk is a small slice of a document used as a retrieval unit.", "label": "embed"},
    {"prompt": "What is reranking?", "completion": "Reranking reorders retrieved chunks by relevance before generation.", "label": "rag"},
    {"prompt": "What is tokenization?", "completion": "Tokenization splits text into subword units the model understands.", "label": "embed"},
]


def chat_template(ex):
    return f"<|user|>\n{ex['prompt']}\n<|assistant|>\n{ex['completion']}"


def stratified_split(examples, val_frac=0.25):
    groups = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex["label"], []).append(i)
    train, val = [], []
    for label, idxs in sorted(groups.items()):
        k = max(1, round(len(idxs) * val_frac))
        val.extend(idxs[:k])
        train.extend(idxs[k:])
    return [examples[i] for i in train], [examples[i] for i in val]


def main():
    print("=== Approach 3: dataset preparation (offline) ===")
    formatted = [chat_template(e) for e in RAW]
    lens = [len(t.split()) for t in formatted]
    print(f"examples={len(RAW)}  avg_words={sum(lens)/len(lens):.1f}  min={min(lens)}  max={max(lens)}")
    train, val = stratified_split(RAW)
    print(f"split -> train={len(train)}  val={len(val)}")
    assert len(train) + len(val) == len(RAW), "split must cover all examples"
    tset = {id(e) for e in train}
    vset = {id(e) for e in val}
    assert not (tset & vset), "no leakage between train and val"
    assert all("<|assistant|>" in t for t in formatted), "template must mark the completion"
    tl = {e["label"] for e in train}
    vl = {e["label"] for e in val}
    assert vl == tl, "stratified split keeps every label in both splits"
    print("sample:", formatted[0].replace(chr(10), " | ")[:70])
    print("self-check OK: templated, split, no leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
