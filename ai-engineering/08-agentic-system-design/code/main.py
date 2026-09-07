import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The Autonomous Research Agent (Week 8 weekly build).

LangGraph loop: decide (LLM -> tool call OR final) -> act (search/fetch over a
local corpus) -> observe -> back to decide. Guards: max iterations, tool-input
validation, graceful stop. The model can re-query with better args (self-correct).
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
    {"id": "r1", "title": "Hybrid Search", "text": "Hybrid search fuses BM25 keyword matching with dense vector similarity using Reciprocal Rank Fusion (RRF). RRF is used in production search to combine ranked lists robustly."},
    {"id": "r2", "title": "Reranking", "text": "Cross-encoder reranking scores the query-document pair jointly for higher precision than bi-encoders."},
    {"id": "r3", "title": "Evals", "text": "RAGAS measures faithfulness, answer relevancy, and context recall/precision."},
]


def tool_search(query):
    q = query.lower()
    hits = [f"[{d['id']}] {d['title']}: {d['text'][:90]}" for d in CORPUS if any(t in d["text"].lower() for t in q.split())]
    return "\n".join(hits) if hits else "(no matches)"


def tool_fetch(doc_id):
    for d in CORPUS:
        if d["id"] == doc_id:
            return f"[{d['id']}] {d['title']}: {d['text']}"
    return f"(not found: {doc_id!r})"


TOOLS = {
    "search": {"desc": "Search the corpus. args: {query: str}", "fn": tool_search, "req": "query"},
    "fetch": {"desc": "Fetch a full document by id. args: {id: str}", "fn": tool_fetch, "req": "id"},
}
MAX_STEPS = 4


def build_graph():
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import HumanMessage, SystemMessage

    class State(TypedDict):
        question: str
        observations: list
        pending_tool: str
        pending_args: dict
        answer: str
        steps: int

    llm = get_llm()
    tool_list = "\n".join(f"- {name}: {t['desc']}" for name, t in TOOLS.items())

    def decide(state):
        obs = "\n".join(state["observations"]) or "(none yet)"
        prompt = (
            "You are an autonomous research agent. Use tools to answer, then stop.\n"
            f"Available tools:\n{tool_list}\n\n"
            f"Question: {state['question']}\nObservations so far:\n{obs}\n\n"
            'Respond with ONLY JSON: either {"action":"<tool>","args":{...}} or {"action":"final","answer":"..."}.'
        )
        resp = llm.invoke([SystemMessage("You are a precise agent. Output only valid JSON."), HumanMessage(prompt)])
        txt = resp.content.strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        try:
            decision = json.loads(m.group(0)) if m else {}
        except Exception:
            decision = {}
        action = decision.get("action")
        steps = state["steps"] + 1
        if action in TOOLS:
            args = decision.get("args", {}) or {}
            req = TOOLS[action]["req"]
            if not isinstance(args, dict) or not str(args.get(req, "")).strip():
                return {"observations": state["observations"] + [f"(invalid {action} args; need '{req}')"], "steps": steps}
            return {"pending_tool": action, "pending_args": args, "steps": steps}
        if action == "final":
            return {"answer": str(decision.get("answer", "")), "steps": steps}
        if steps >= MAX_STEPS:
            return {"answer": "Stopped at step limit. " + obs, "steps": steps}
        return {"observations": state["observations"] + ["(unrecognized action; use search/fetch/final)"], "steps": steps}

    def act(state):
        name = state["pending_tool"]
        args = state["pending_args"]
        req = TOOLS[name]["req"]
        result = TOOLS[name]["fn"](str(args.get(req, "")))
        return {"observations": state["observations"] + [f"{name}({args.get(req)}) -> {result}"], "pending_tool": "", "pending_args": {}}

    def route(state):
        if state["pending_tool"] and state["steps"] < MAX_STEPS:
            return "act"
        if state["answer"]:
            return END
        if state["steps"] >= MAX_STEPS:
            return END
        return "decide"

    g = StateGraph(State)
    g.add_node("decide", decide)
    g.add_node("act", act)
    g.add_edge(START, "decide")
    g.add_conditional_edges("decide", route)
    g.add_edge("act", "decide")
    return g.compile()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    q = "What is RRF and where is it used?" if args.selftest else "Explain hybrid search."
    print("=== Autonomous Research Agent (LangGraph loop) ===")
    print(f"query = {q!r}\n")
    try:
        graph = build_graph()
        result = graph.invoke({"question": q, "observations": [], "pending_tool": "", "pending_args": {}, "answer": "", "steps": 0})
    except Exception as e:
        print(f"[agent] unavailable ({type(e).__name__}: {e})")
        print("Hint: needs a configured LLM (Yolo-Auto). Tools run offline; the loop needs the model.")
        return 0
    print(f"steps taken: {result['steps']}")
    for o in result["observations"]:
        print(f"  obs> {o[:110]}")
    print(f"\nANSWER:\n{result['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
