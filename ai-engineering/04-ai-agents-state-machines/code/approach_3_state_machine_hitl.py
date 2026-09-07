import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 3 - rule-based finite state machine with a human-in-the-loop gate.

States: COLLECTING -> VERIFYING -> AWAITING_APPROVAL -> ESCALATED | DONE.
No LLM: deterministic, fully offline. Demonstrates explicit control flow + HITL.
"""
import argparse

TICKET = {"id": "T-555", "issue": "damaged item on delivery", "order": "A-1003", "refund_requested": True}


def collect(state):
    missing = [k for k in ("id", "issue", "order") if not state.get(k)]
    if missing:
        return "COLLECTING", f"missing fields: {missing}"
    return "VERIFYING", "fields complete"


def verify(state):
    if state.get("refund_requested"):
        return "AWAITING_APPROVAL", "refund candidate; needs human approval"
    return "DONE", "no action needed"


def approve(state, human_decision):
    if human_decision == "approve":
        return "ESCALATED", "refund approved by human"
    return "DONE", "human declined; ticket closed"


def run(ticket, human_decision="approve"):
    state = dict(ticket)
    trace = []
    nxt, note = collect(state)
    trace.append((nxt, note))
    if nxt != "VERIFYING":
        return trace
    nxt, note = verify(state)
    trace.append((nxt, note))
    if nxt == "AWAITING_APPROVAL":
        nxt, note = approve(state, human_decision)
        trace.append((nxt, note))
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", choices=["approve", "decline"], default="approve")
    args = ap.parse_args()
    print("=== Approach 3: FSM with human-in-the-loop (offline) ===")
    print(f"ticket = {TICKET}\nhuman decision = {args.decision!r}\n")
    trace = run(TICKET, args.decision)
    for i, (s, n) in enumerate(trace, 1):
        print(f"  {i}. -> {s:20} ({n})")
    final = trace[-1][0]
    expected = "ESCALATED" if args.decision == "approve" else "DONE"
    assert final == expected, f"expected {expected}, got {final}"
    print(f"\nself-check OK: final state {final} matches decision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
