import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - deployment strategy decision matrix (offline).

Scores batch vs streaming and on-prem vs cloud across criteria per scenario;
recommends one. Teaches which deployment fits which task.
"""

SCENARIOS = {
    "interactive_chatbot": {
        "options": {"batch": [0.3, 0.9, 0.9, 0.5], "streaming": [0.9, 0.6, 0.5, 0.7]},
        "weights": [0.4, 0.2, 0.25, 0.15],
        "expect": "streaming",
    },
    "overnight_report_batch": {
        "options": {"batch": [0.7, 0.9, 0.9, 0.9], "streaming": [0.4, 0.5, 0.6, 0.4]},
        "weights": [0.1, 0.3, 0.35, 0.25],
        "expect": "batch",
    },
    "regulated_finance_data": {
        "options": {"on_prem": [0.95, 0.5, 0.7, 0.6], "cloud": [0.5, 0.9, 0.9, 0.9]},
        "weights": [0.6, 0.15, 0.1, 0.15],
        "expect": "on_prem",
    },
}


def recommend(name):
    s = SCENARIOS[name]
    w = s["weights"]
    tot = {o: sum(a * b for a, b in zip(sc, w)) for o, sc in s["options"].items()}
    return max(tot, key=tot.get), tot


def main():
    print("=== Benchmark: deployment strategies (offline) ===")
    for name, s in SCENARIOS.items():
        best, tot = recommend(name)
        print(f"\n[{name}]")
        for o in s["options"]:
            print(f"  {o:10}: {tot[o]:.3f}")
        print(f"  -> recommend: {best}")
        assert best == s["expect"], f"{name}: expected {s['expect']}, got {best}"
    print("\nself-check OK: streaming for interactive, batch for overnight, on-prem for regulated data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
