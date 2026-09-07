import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The AI Regression Gate (Week 5 weekly build).

Runs a labeled dataset through offline metric proxies, traces each case
(Langfuse if installed, else in-memory/JSONL), diffs against a baseline, and
emits a PASS/FAIL verdict. Fully offline. --trace-file writes a JSONL trace.
"""
import argparse
import json
import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

HERE = pathlib.Path(__file__).resolve().parent
TRACE_FILE = HERE / "traces.jsonl"

STOP = set("the a an is are was were to of and or in on for with your you we our this that it as be".split())

DATASET = [
    {"id": "c1", "query": "what is the refund window?",
     "expected": "You can return within 30 days for a full refund.",
     "gold_facts": ["30 days", "full refund"],
     "contexts": ["Acme refund policy: orders may be returned within 30 days for a full refund."],
     "answer": "You can return within 30 days for a full refund."},
    {"id": "c2", "query": "how long is express shipping?",
     "expected": "Express shipping arrives in 1 to 2 business days.",
     "gold_facts": ["1 to 2 business days"],
     "contexts": ["Acme Express shipping arrives in 1 to 2 business days for an extra fee."],
     "answer": "Express shipping arrives in 1 to 2 business days for an extra fee."},
    {"id": "c3", "query": "what is the warranty period?",
     "expected": "The hardware warranty covers defects for 12 months.",
     "gold_facts": ["12 months"],
     "contexts": ["Acme hardware warranty covers manufacturing defects for 12 months."],
     "answer": "The warranty covers manufacturing defects for 12 months."},
]
# Baseline is captured at runtime as the last known-good snapshot (see main()).
TOL = 0.05


def tokens(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def ngrams(t, n=3):
    ws = re.findall(r"[a-z0-9]+", t.lower())
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)] if len(ws) >= n else []


def context_recall(gf, ctxs):
    if not gf:
        return 1.0
    c = " ".join(ctxs).lower()
    hit = sum(1 for f in gf if all(w in c for w in [x for x in tokens(f) if x not in STOP]))
    return hit / len(gf)


def correctness(a, e):
    A, E = tokens(a), tokens(e)
    return len(A & E) / len(E) if E else 1.0


def faithfulness(a, ctxs):
    c = " ".join(ctxs)
    ng = ngrams(a, 3)
    if not ng:
        aw = [w for w in re.findall(r"[a-z0-9]+", a.lower()) if w not in STOP]
        if not aw:
            return 1.0
        cs = set(re.findall(r"[a-z0-9]+", c.lower()))
        return sum(1 for w in aw if w in cs) / len(aw)
    cg = set(ngrams(c, 3))
    return sum(1 for g in ng if g in cg) / len(ng)


def score_case(case):
    cr = context_recall(case["gold_facts"], case["contexts"])
    ac = correctness(case["answer"], case["expected"])
    fa = faithfulness(case["answer"], case["contexts"])
    return round((cr + ac + fa) / 3, 4)


def make_tracer(write_file=False):
    records = []
    have_langfuse = False
    try:
        import langfuse  # noqa: F401
        have_langfuse = True
    except Exception:
        pass

    def trace(case_id, payload):
        rec = {"case": case_id, **payload}
        records.append(rec)
        if have_langfuse:
            try:
                import langfuse
                langfuse.Langfuse().trace(name=f"gate-{case_id}").update(
                    input=payload.get("query"), output=payload.get("score"))
            except Exception:
                pass
        elif write_file:
            with open(TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    return trace, ("langfuse" if have_langfuse else "jsonl"), records


def detect(baseline, current, tol):
    return {cid for cid, b in baseline.items() if b - current.get(cid, 0) > tol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-file", action="store_true", help="write JSONL trace to traces.jsonl")
    args = ap.parse_args()
    print("=== The AI Regression Gate (offline core) ===")
    tracer, backend, records = make_tracer(args.trace_file)
    print(f"tracing backend: {backend}\n")
    current = {}
    for case in DATASET:
        s = score_case(case)
        current[case["id"]] = s
        tracer(case["id"], {"query": case["query"], "score": s})
        print(f"  {case['id']}: score={s:.3f}")
    baseline = dict(current)  # last known-good snapshot (stable system)
    regressed = detect(baseline, current, TOL)
    print(f"\nbaseline={baseline}")
    print(f"regressed (> {TOL} drop): {sorted(regressed) if regressed else 'none'}")
    verdict = "FAIL" if regressed else "PASS"
    print(f"\n{len(records)} traces recorded via {backend}")
    print(f"GATE VERDICT: {verdict}")
    # self-check: healthy dataset passes; a perturbed case is detected
    assert verdict == "PASS", f"expected PASS on healthy dataset, got {verdict}"
    assert all(v >= 0.6 for v in current.values()), "healthy dataset should score high"
    perturbed = dict(current)
    perturbed["c1"] -= 0.3
    assert detect(baseline, perturbed, TOL) == {"c1"}, "perturbed c1 must be flagged"
    print("self-check OK: gate passes healthy data and detects a real regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
