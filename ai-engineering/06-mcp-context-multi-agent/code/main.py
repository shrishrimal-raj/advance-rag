import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The Research Assistant (Week 6 weekly build).

LangGraph StateGraph multi-agent pipeline: plan -> research -> synthesize.
plan/synthesize use the cloud LLM (2 calls under --selftest); research is a
deterministic tool pass over a local corpus (MCP tools added if available).
"""
import argparse
import json
import pathlib
import re
import sys
from typing import TypedDict

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm  # noqa: E402

CORPUS = [
    {"id": "r1", "topic": "retrieval", "text": "Hybrid search combines BM25 keyword matching with dense vector similarity, fused via RRF."},
    {"id": "r2", "topic": "reranking", "text": "Cross-encoders score the query-document pair directly and are more accurate than bi-encoders but slower."},
    {"id": "r3", "topic": "evaluation", "text": "RAGAS measures faithfulness, answer relevancy, and context recall/precision for RAG systems."},
    {"id": "r4", "topic": "agents", "text": "Multi-agent systems split work across specialized agents to keep each context small and focused."},
]


def search_corpus(query):
    q = query.lower()
    hits = [f"[{d['id']}] {d['text']}" for d in CORPUS if any(t in d["text"].lower() for t in q.split())]
    return "\n".join(hits) if hits else "(no matches)"


def mcp_tools_available():
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False


def build_graph():
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import HumanMessage, SystemMessage

    class State(TypedDict):
        question: str
        plan: list
        findings: list
        answer: str

    llm = get_llm()

    def plan(state):
        resp = llm.invoke([
            SystemMessage("You are a research planner. Return ONLY a JSON array of 1-3 short sub-questions."),
            HumanMessage(state["question"]),
        ])
        txt = resp.content.strip()
        m = re.search(r"\[.*\]", txt, re.DOTALL)
        subs = []
        if m:
            try:
                subs = json.loads(m.group(0))
            except Exception:
                subs = []
        if not subs:
            subs = [line.strip("- ").strip() for line in txt.splitlines() if line.strip()]
        return {"plan": subs or [state["question"]]}

    def research(state):
        out = []
        for sq in state["plan"]:
            out.append(f"Q: {sq}\n{search_corpus(sq)}")
        return {"findings": out}

    def synthesize(state):
        findings = "\n\n".join(state["findings"])
        resp = llm.invoke([
            SystemMessage("You are a research synthesizer. Answer using ONLY the findings. Cite [r#] ids."),
            HumanMessage(f"Question: {state['question']}\n\nFindings:\n{findings}"),
        ])
        return {"answer": resp.content}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="How do hybrid search and reranking improve RAG quality?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    q = "How do hybrid search and reranking improve RAG quality?" if args.selftest else args.query
    print("=== Research Assistant (LangGraph multi-agent) ===")
    print(f"query = {q!r}")
    print(f"MCP tools available: {mcp_tools_available()} (local corpus used either way)\n")
    try:
        graph = build_graph()
        result = graph.invoke({"question": q, "plan": [], "findings": [], "answer": ""})
    except Exception as e:
        print(f"[research assistant] unavailable ({type(e).__name__}: {e})")
        print("Hint: needs a configured LLM (Yolo-Auto). Tools run offline; planning/synthesis need the model.")
        return 0
    print(f"PLAN: {result['plan']}")
    print(f"\nANSWER:\n{result['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
