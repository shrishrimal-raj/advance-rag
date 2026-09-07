import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 2 - real RAGAS metrics (opt-in LLM).

Builds SingleTurnSamples and lists RAGAS metrics. Default is a dry-run that
constructs everything offline and prints the plan. --run calls ragas.evaluate
(needs an LLM configured for RAGAS).
"""
import argparse
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

SAMPLES = [
    {"user_input": "What is the refund window?",
     "retrieved_contexts": ["Acme refund policy: orders may be returned within 30 days for a full refund."],
     "response": "You can return within 30 days for a full refund.",
     "reference": "30 days, full refund"},
]

METRIC_NAMES = ["Faithfulness", "AnswerRelevancy", "ContextRecall"]


def build_samples():
    from ragas.dataset_schema import SingleTurnSample
    return [SingleTurnSample(**s) for s in SAMPLES]


def build_metrics(llm):
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall
    return [Faithfulness(llm=llm), AnswerRelevancy(llm=llm), ContextRecall(llm=llm)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="compute live scores (needs LLM)")
    args = ap.parse_args()
    print("=== Approach 2: RAGAS metrics ===")
    try:
        samples = build_samples()
    except Exception as e:
        print(f"[ragas] unavailable ({type(e).__name__}: {e})")
        print("Hint: check ragas.dataset_schema.SingleTurnSample field names.")
        return 0
    print(f"samples: {len(samples)}   metrics: {METRIC_NAMES}")
    if args.run:
        try:
            from ragas import evaluate
            from shared.config import get_llm
            metrics = build_metrics(get_llm())
            result = evaluate(samples, metrics=metrics)
            print(result.to_pandas().to_string())
        except Exception as e:
            print(f"[ragas.evaluate] failed ({type(e).__name__}: {e})")
            print("Hint: --run needs an LLM; RAGAS may require wrapping it (ragas.llms.LangchainLLM). Default dry-run stays offline.")
        return 0
    print("mode: dry-run (no LLM). Re-run with --run to compute live scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
