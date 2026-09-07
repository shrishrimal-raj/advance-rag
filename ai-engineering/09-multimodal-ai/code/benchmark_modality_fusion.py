import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - multimodal fusion strategy decision matrix (offline).

Scores unimodal vs early-fusion vs late-fusion across criteria per scenario and
recommends one. Teaches which fusion fits which task.
"""
STRATS = ["unimodal", "early_fusion", "late_fusion"]

SCENARIOS = {
    "invoice_doc_understanding": {
        "scores": {"unimodal": [0.4, 0.9, 0.9, 0.9], "early_fusion": [0.7, 0.5, 0.6, 0.4], "late_fusion": [0.85, 0.7, 0.7, 0.7]},
        "weights": [0.4, 0.2, 0.2, 0.2],
        "expect": "late_fusion",
    },
    "paired_image_text_search": {
        "scores": {"unimodal": [0.3, 0.9, 0.9, 0.9], "early_fusion": [0.95, 0.5, 0.6, 0.4], "late_fusion": [0.7, 0.7, 0.7, 0.6]},
        "weights": [0.5, 0.15, 0.15, 0.2],
        "expect": "early_fusion",
    },
}


def recommend(name):
    s = SCENARIOS[name]
    w = s["weights"]
    tot = {p: sum(a * b for a, b in zip(sc, w)) for p, sc in s["scores"].items()}
    return max(tot, key=tot.get), tot


def main():
    print("=== Benchmark: multimodal fusion strategy (offline) ===")
    print("criteria: accuracy, cost, latency, data_needs")
    for name, s in SCENARIOS.items():
        best, tot = recommend(name)
        print(f"\n[{name}]")
        for p in STRATS:
            print(f"  {p:12}: {tot[p]:.3f}")
        print(f"  -> recommend: {best}")
        assert best == s["expect"], f"{name}: expected {s['expect']}, got {best}"
    print("\nself-check OK: late fusion for modular doc tasks, early fusion for tightly-paired data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
