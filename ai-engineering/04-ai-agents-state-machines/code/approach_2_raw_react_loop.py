import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 2 - raw ReAct/tool-calling loop, no framework.

Shows the mechanism LangGraph hides: bind tools to the LLM, then loop
invoke -> execute tool_calls -> append ToolMessages -> repeat until final answer.
"""
import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm  # noqa: E402

DOCS = {
    "refund": "Acme refund policy: orders may be returned within 30 days for a full refund.",
    "warranty": "Acme hardware warranty covers manufacturing defects for 12 months.",
}
ORDERS = {
    "A-1001": {"status": "shipped", "eta": "2 days"},
    "A-1002": {"status": "processing", "eta": "5 days"},
    "A-1003": {"status": "delivered", "eta": "delivered"},
}


def search_docs(query: str) -> str:
    """Search the Acme knowledge base for policy/help text matching the query."""
    q = query.lower()
    hits = [f"{k}: {v}" for k, v in DOCS.items() if any(t in v.lower() for t in q.split())]
    return "\n".join(hits) if hits else "No matching docs."


def lookup_order(order_id: str) -> str:
    """Look up an order by its ID (e.g. A-1001) and return status/eta."""
    o = ORDERS.get(order_id)
    return f"Order {order_id}: {o}" if o else f"Order {order_id} not found."


def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent with a reason."""
    return f"[ESCALATED to human] reason: {reason}"


TOOLS = {"search_docs": search_docs, "lookup_order": lookup_order, "escalate_to_human": escalate_to_human}


def react(query, max_steps=6):
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    llm = get_llm().bind_tools([search_docs, lookup_order, escalate_to_human])
    msgs = [SystemMessage("You are an Acme support agent. Use the tools to answer."), HumanMessage(query)]
    for step in range(1, max_steps + 1):
        resp = llm.invoke(msgs)
        msgs.append(resp)
        if getattr(resp, "tool_calls", None):
            for tc in resp.tool_calls:
                fn = TOOLS.get(tc["name"])
                out = fn(**tc["args"]) if fn else f"unknown tool {tc['name']}"
                print(f"  [step {step}] {tc['name']}({tc['args']}) -> {str(out)[:80]}")
                msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
            continue
        return resp.content
    return "(max steps reached)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    q = "What's the status of order A-1002?" if args.selftest else "Check order A-1003 and explain the warranty."
    print("=== Approach 2: raw ReAct loop (no framework) ===")
    print(f"query = {q!r}")
    try:
        print("\nANSWER:\n" + react(q))
    except Exception as e:
        print(f"[react] unavailable ({type(e).__name__}: {e})")
        print("Hint: needs a configured LLM.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
