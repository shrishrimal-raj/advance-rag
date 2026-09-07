import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - single-agent vs multi-agent context load (offline).

A task with several sub-skills. Single agent: one prompt holds ALL role
instructions + ALL data. Multi-agent: each node holds only its own slice. We
measure per-context token load (word proxy) and show multi-agent keeps each
context smaller than the single-agent monolith.
"""

ROLE_INSTRUCTIONS = {
    "planner": "Break the question into sub-questions. Be precise.",
    "researcher": "Answer each sub-question using only provided sources. Cite.",
    "synthesizer": "Merge findings into one cited answer. No new claims.",
}
DATA_SLICES = {
    "planner": ["question"],
    "researcher": ["question", "sources"],
    "synthesizer": ["findings"],
}
ITEM_COST = {"question": 10, "sources": 60, "findings": 40}
INSTR_COST = {k: len(v.split()) for k, v in ROLE_INSTRUCTIONS.items()}


def single_agent_load():
    return sum(INSTR_COST.values()) + sum(ITEM_COST.values())


def multi_agent_loads():
    return {role: INSTR_COST[role] + sum(ITEM_COST[i] for i in items) for role, items in DATA_SLICES.items()}


def main():
    print("=== Benchmark: single vs multi-agent context load (offline) ===")
    single = single_agent_load()
    multi = multi_agent_loads()
    print(f"single-agent total context: {single} tokens (one monolithic prompt)")
    peak = 0
    for role, load in multi.items():
        print(f"  multi-agent [{role:12}]: {load} tokens")
        peak = max(peak, load)
    print(f"\nmulti-agent peak per-node context: {peak} tokens")
    assert peak < single, "multi-agent peak node must be smaller than the single-agent monolith"
    assert all(v < single for v in multi.values()), "every node smaller than monolith"
    print("self-check OK: multi-agent isolates context, keeping each node lean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
