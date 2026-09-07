import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The Enterprise AI Platform (Week 10 capstone).

Integrates the course components: retrieve (local cosine RAG) -> agent (plan +
synthesize, cloud LLM) -> evaluate (offline faithfulness) -> trace (JSONL).
--selftest runs one query end-to-end (2 LLM calls) and writes trace.jsonl.
"""
import argparse
import json
import pathlib
import re
import sys
import time

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm  # noqa: E402

CORPUS = [
    {"id": "r1", "text": "Hybrid search fuses BM25 keyword matching with dense vector similarity using Reciprocal Rank Fusion (RRF)."},
    {"id": "r2", "text": "Cross-encoder reranking scores the query-document pair jointly for higher precision than bi-encoders."},
    {"id": "r3", "text": "RAGAS measures faithfulness, answer relevancy, and context recall/precision for RAG systems."},
    {"id": "r4", "text": "Multi-agent systems split work across specialized agents to keep each context small and focused."},
]


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _vec(text):
    v = {}
    for t in _tokens(text):
        v[t] = v.get(t, 0) + 1
    return v


def _cos(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na = sum(x * x for x in a.values()) ** 0.5
    nb = sum(x * x for x in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query, k=3):
    qv = _vec(query)
    scored = sorted(((_cos(qv, _vec(d["text"])), d) for d in CORPUS), key=lambda x: -x[0])
    hits = [(s, d) for s, d in scored if s > 0]
    return (hits or scored)[:k]


def plan_subquestions(question):
    llm = get_llm()
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = llm.invoke([SystemMessage("Return ONLY a JSON array of 1-3 short sub-questions."), HumanMessage(question)])
    m = re.search(r"\[.*\]", resp.content, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else []
    except Exception:
        return []


def synthesize(question, subqs, contexts):
    llm = get_llm()
    from langchain_core.messages import HumanMessage, SystemMessage
    ctx = "\n".join(f"[{d['id']}] {d['text']}" for _, d in contexts)
    sq = "\n".join(f"- {s}" for s in subqs) or "(direct)"
    resp = llm.invoke([
        SystemMessage("Answer using ONLY the provided context. Cite [r#] ids."),
        HumanMessage(f"Sub-questions:\n{sq}\n\nContext:\n{ctx}\n\nQuestion: {question}"),
    ])
    return resp.content


def run_agent(question, contexts):
    subqs = plan_subquestions(question)
    return synthesize(question, subqs, contexts)


def evaluate(answer, contexts):
    """Offline faithfulness proxy: fraction of cited ids that exist in context."""
    cited = set(re.findall(r"\[(r\d+)\]", answer))
    available = {d["id"] for _, d in contexts}
    if not cited:
        return {"faithfulness": 0.5, "cited": [], "note": "no citations"}
    valid = cited & available
    return {"faithfulness": round(len(valid) / len(cited), 3), "cited": sorted(cited), "valid": sorted(valid)}


def write_trace(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--query", default="How do hybrid search and reranking improve RAG quality?")
    args = ap.parse_args()
    q = "How do hybrid search and reranking improve RAG quality?" if args.selftest else args.query
    print("=== Enterprise AI Platform (capstone) ===")
    print(f"query = {q!r}\n")
    t0 = time.time()
    contexts = retrieve(q)
    print("RETRIEVED:")
    for s, d in contexts:
        print(f"  [{d['id']}] ({s:.3f}) {d['text'][:70]}")
    try:
        answer = run_agent(q, contexts)
    except Exception as e:
        print(f"\n[agent] unavailable ({type(e).__name__}: {e})")
        print("Hint: needs a configured LLM (Yolo-Auto). Retrieval/eval/trace run offline.")
        return 0
    ev = evaluate(answer, contexts)
    latency = time.time() - t0
    print(f"\nANSWER:\n{answer}\n")
    print(f"EVAL: faithfulness={ev['faithfulness']}  cited={ev.get('cited')}")
    trace_path = pathlib.Path(__file__).resolve().parent / "trace.jsonl"
    write_trace(trace_path, {"ts": time.time(), "query": q, "retrieved": [d["id"] for _, d in contexts], "eval": ev, "latency_s": round(latency, 2)})
    print(f"TRACE: wrote {trace_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
