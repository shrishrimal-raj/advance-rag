import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 3 - planning + memory from scratch (offline).

A rule-based planner decomposes a goal into subtasks, executes them, stores results
in a memory store, and retrieves relevant memory for a follow-up. No model needed.
"""

MEMORY = {}


def plan(goal):
    if "report" in goal.lower():
        return ["gather_data", "analyze", "summarize"]
    return ["do_goal"]


def execute(step, ctx):
    if step == "gather_data":
        return {"records": [10, 20, 30]}
    if step == "analyze":
        recs = ctx.get("records", [])
        return {"total": sum(recs), "count": len(recs)}
    if step == "summarize":
        a = ctx.get("analysis", {})
        return {"summary": f"total={a.get('total')} over {a.get('count')} records"}
    return {"done": True}


def run(goal):
    steps = plan(goal)
    ctx = {}
    trace = []
    for s in steps:
        res = execute(s, ctx)
        MEMORY[s] = res
        ctx[s] = res
        if s == "gather_data":
            ctx["records"] = res["records"]
        if s == "analyze":
            ctx["analysis"] = res
        trace.append((s, res))
    return steps, trace


def recall(keyword):
    return {k: v for k, v in MEMORY.items() if keyword.lower() in k.lower()}


def main():
    print("=== Approach 3: planning + memory (offline) ===")
    steps, trace = run("Build a report")
    print(f"plan: {steps}")
    for s, r in trace:
        print(f"  {s} -> {r}")
    followup = recall("analy")
    print(f"recall('analy') -> {followup}")
    assert steps == ["gather_data", "analyze", "summarize"], "planner must decompose the report goal"
    assert trace[-1][1]["summary"] == "total=60 over 3 records", "pipeline must flow data between steps"
    assert recall("analy") == {"analyze": {"total": 60, "count": 3}}, "memory must recall prior results"
    print("self-check OK: plans, executes with shared context, recalls from memory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
