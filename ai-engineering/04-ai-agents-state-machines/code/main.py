import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The Customer Support Agent (Week 4 weekly build).

A LangGraph ReAct agent with three tools: search_docs, lookup_order,
escalate_to_human. Uses the cloud LLM. Run --selftest for one end-to-end query.
"""
import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm  # noqa: E402

DOCS = {
    "refund": "Acme refund policy: orders may be returned within 30 days for a full refund.",
    "shipping": "Acme Express shipping arrives in 1 to 2 business days for an extra fee.",
    "warranty": "Acme hardware warranty covers manufacturing defects for 12 months.",
}
ORDERS = {
    "A-1001": {"status": "shipped", "eta": "2 days", "item": "Widget Pro"},
    "A-1002": {"status": "processing", "eta": "5 days", "item": "Gadget X"},
    "A-1003": {"status": "delivered", "eta": "delivered", "item": "Doohickey"},
}


def search_docs(query: str) -> str:
    """Search the Acme knowledge base for policy/help text matching the query."""
    q = query.lower()
    hits = [f"{k}: {v}" for k, v in DOCS.items() if any(t in v.lower() for t in q.split())]
    return "\n".join(hits) if hits else "No matching docs."


def lookup_order(order_id: str) -> str:
    """Look up an order by its ID (e.g. A-1001) and return status/eta/item."""
    o = ORDERS.get(order_id)
    return f"Order {order_id}: {o}" if o else f"Order {order_id} not found."


def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent with a reason."""
    return f"[ESCALATED to human] reason: {reason}"


def build_agent():
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(get_llm(), [search_docs, lookup_order, escalate_to_human])


def run(query):
    agent = build_agent()
    result = agent.invoke({"messages": [("user", query)]})
    return result["messages"][-1].content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="What's the status of order A-1001 and what's your refund policy?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    q = "What's the status of order A-1001?" if args.selftest else args.query
    print("=== Customer Support Agent (LangGraph ReAct) ===")
    print(f"query = {q!r}")
    try:
        answer = run(q)
    except Exception as e:
        print(f"[agent] unavailable ({type(e).__name__}: {e})")
        print("Hint: needs a configured LLM (Yolo-Auto). Tools work offline; the loop needs the model.")
        return 0
    print(f"\nANSWER:\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
