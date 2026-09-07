import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - fine-tuning vs RAG decision matrix (offline).

Scores both approaches across criteria for a given scenario and recommends one.
Teaches the core trade-off: RAG for changing knowledge, fine-tune for behavior.
"""

SCENARIOS = {
    "legal_qa_changing_laws": {
        "scores": {"rag": [0.9, 0.8, 0.8, 0.4, 0.9], "finetune": [0.2, 0.3, 0.9, 0.7, 0.5]},
        "weights": [0.35, 0.15, 0.15, 0.15, 0.20],
        "expect": "rag",
    },
    "strict_json_output_style": {
        "scores": {"rag": [0.5, 0.8, 0.8, 0.2, 0.6], "finetune": [0.5, 0.4, 0.9, 0.95, 0.7]},
        "weights": [0.10, 0.15, 0.15, 0.45, 0.15],
        "expect": "finetune",
    },
}
CRITERIA = ["knowledge_freshness", "cost", "latency", "format_control", "data_availability"]


def recommend(name):
    s = SCENARIOS[name]
    w = s["weights"]
    tot = {ap: sum(a * b for a, b in zip(sc, w)) for ap, sc in s["scores"].items()}
    return max(tot, key=tot.get), tot


def main():
    print("=== Benchmark: fine-tune vs RAG decision matrix (offline) ===")
    print("criteria:", ", ".join(CRITERIA))
    for name, s in SCENARIOS.items():
        best, tot = recommend(name)
        print(f"\n[{name}]")
        for ap in ("rag", "finetune"):
            print(f"  {ap:9}: {tot[ap]:.3f}")
        print(f"  -> recommend: {best}")
        assert best == s["expect"], f"{name}: expected {s['expect']}, got {best}"
    print("\nself-check OK: RAG wins for fresh knowledge, fine-tune wins for format/behavior")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
