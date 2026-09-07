import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - agent design patterns decision matrix (offline).

Scores ReAct vs plan-and-execute vs reflection across criteria per scenario and
recommends one. Teaches which pattern fits which task.
"""

PATTERNS = ["react", "plan_execute", "reflection"]

SCENARIOS = {
    "open_ended_research": {
        "scores": {"react": [0.5, 0.7, 0.5, 0.95], "plan_execute": [0.7, 0.7, 0.8, 0.4], "reflection": [0.4, 0.8, 0.4, 0.7]},
        "weights": [0.25, 0.2, 0.2, 0.35],
        "expect": "react",
    },
    "known_procedure_invoice": {
        "scores": {"react": [0.4, 0.6, 0.5, 0.3], "plan_execute": [0.85, 0.85, 0.85, 0.5], "reflection": [0.5, 0.8, 0.5, 0.4]},
        "weights": [0.3, 0.3, 0.25, 0.15],
        "expect": "plan_execute",
    },
}


def recommend(name):
    s = SCENARIOS[name]
    w = s["weights"]
    tot = {p: sum(a * b for a, b in zip(sc, w)) for p, sc in s["scores"].items()}
    return max(tot, key=tot.get), tot


def main():
    print("=== Benchmark: agent design patterns (offline) ===")
    print("criteria: cost, reliability, latency, adaptability")
    for name, s in SCENARIOS.items():
        best, tot = recommend(name)
        print(f"\n[{name}]")
        for p in PATTERNS:
            print(f"  {p:13}: {tot[p]:.3f}")
        print(f"  -> recommend: {best}")
        assert best == s["expect"], f"{name}: expected {s['expect']}, got {best}"
    print("\nself-check OK: ReAct for open-ended, plan-and-execute for known procedures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
