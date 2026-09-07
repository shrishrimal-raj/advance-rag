import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 3 - context window management from scratch (offline).

fit_context(system, messages, budget): always keep system + latest turn; fill the
remaining budget with the most recent history; summarize evicted turns into a
running note that is truncated to fit. Token proxy = word count. Self-check.
"""


def tokens(text):
    return len(text.split())


def fit_context(system, messages, budget):
    """messages: list of {'role','content'} oldest->newest. Returns (parts, evicted, total)."""
    sys_t = tokens(system)
    latest = messages[-1] if messages else None
    latest_t = tokens(latest["content"]) if latest else 0
    base = sys_t + latest_t
    remaining = budget - base
    kept, evicted = [], []
    if remaining >= 0:
        for m in reversed(messages[:-1]):
            t = tokens(m["content"])
            if t <= remaining:
                kept.append(m)
                remaining -= t
            else:
                evicted.append(m)
        kept.reverse()
    else:
        evicted = list(messages[:-1])
        kept = []
    parts = [system]
    if evicted:
        kept_t = sum(tokens(m["content"]) for m in kept)
        summary_budget = max(0, budget - sys_t - kept_t - latest_t)
        raw = "Earlier (summarized): " + " | ".join(m["content"] for m in reversed(evicted))
        words = raw.split()
        while words and tokens(" ".join(words)) > summary_budget:
            words.pop()
        summary = " ".join(words)
        if summary:
            parts.append(summary)
    parts.extend(m["content"] for m in kept)
    if latest:
        parts.append(latest["content"])
    total = sum(tokens(p) for p in parts)
    return parts, len(evicted), total


CASES = [
    {
        "system": "You are a concise assistant.",
        "budget": 30,
        "messages": [
            {"role": "user", "content": "Tell me about the refund policy please"},
            {"role": "assistant", "content": "The refund policy allows returns within thirty days for a full refund of the purchase price."},
            {"role": "user", "content": "And the warranty?"},
            {"role": "assistant", "content": "The warranty covers manufacturing defects for twelve months from purchase."},
            {"role": "user", "content": "Summarize both briefly."},
        ],
    },
]


def main():
    print("=== Approach 3: context window management (offline) ===")
    for c in CASES:
        parts, evicted, total = fit_context(c["system"], c["messages"], c["budget"])
        print(f"budget={c['budget']}  evicted={evicted}  final_tokens={total}")
        for i, line in enumerate(parts):
            print(f"  [{i}] {line[:60]}{'...' if len(line) > 60 else ''}")
        assert total <= c["budget"], f"must fit budget: {total} > {c['budget']}"
        assert parts[0] == c["system"], "system prompt must be first"
        assert parts[-1] == c["messages"][-1]["content"], "latest turn must be last"
    print("self-check OK: fits budget, system first, latest last, older turns compacted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
