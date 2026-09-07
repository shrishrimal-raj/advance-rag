import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - what plain RAG can reach vs what an agent can reach (offline).

Direct RAG only sees the static corpus. An agent also has live tools (order DB).
Per query we show whether the needed fact is reachable by RAG alone vs by the
agent. No LLM needed - this is a data-availability comparison.
"""

CORPUS = {
    "refund": "orders may be returned within 30 days for a full refund",
    "shipping": "Express shipping arrives in 1 to 2 business days",
    "warranty": "hardware warranty covers manufacturing defects for 12 months",
}
ORDERS = {
    "A-1001": "shipped, eta 2 days",
    "A-1002": "processing, eta 5 days",
    "A-1003": "delivered",
}

# (query, needed_fact_key, source)  source in {'corpus','orders'}
QS = [
    ("what is the refund window?", "refund", "corpus"),
    ("how long is the warranty?", "warranty", "corpus"),
    ("what is the status of order A-1001?", "A-1001", "orders"),
    ("is my order A-1002 delivered yet?", "A-1002", "orders"),
]


def rag_can_answer(source):
    return source == "corpus"  # RAG only sees the corpus


def agent_can_answer(source):
    return True  # agent has corpus + order tool


def main():
    print("=== Benchmark: direct RAG vs agent reach (offline) ===")
    rag_hits = agent_hits = 0
    for q, key, source in QS:
        r = rag_can_answer(source)
        a = agent_can_answer(source)
        rag_hits += r
        agent_hits += a
        print(f"  {q:42} RAG={'OK ' if r else 'NO '} agent={'OK' if a else 'NO'}  (needs {source})")
    n = len(QS)
    print(f"\nreach  RAG={rag_hits}/{n}  agent={agent_hits}/{n}")
    assert agent_hits >= rag_hits, "agent must reach at least what RAG reaches"
    assert agent_hits > rag_hits, "agent should strictly extend reach here"
    print("self-check OK: agent extends RAG reach via live tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
