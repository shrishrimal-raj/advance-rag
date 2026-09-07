import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Benchmark - regression detection: baseline vs current scores (offline).

Flags test cases whose score dropped beyond a tolerance. Deterministic; self-check.
"""

BASELINE = {"c1": 0.95, "c2": 0.80, "c3": 0.70, "c4": 0.60}
CURRENT = {"c1": 0.94, "c2": 0.79, "c3": 0.40, "c4": 0.61}
TOL = 0.05


def detect_regressions(baseline, current, tol):
    regs = {}
    for cid, base in baseline.items():
        cur = current.get(cid)
        if cur is None:
            regs[cid] = ("missing", base, None)
            continue
        if base - cur > tol:
            regs[cid] = ("drop", base, cur)
    return regs


def gate(regs, critical=("c3",)):
    return "FAIL" if any(cid in regs for cid in critical) else "PASS"


def main():
    print("=== Benchmark: regression detector (offline) ===")
    regs = detect_regressions(BASELINE, CURRENT, TOL)
    for cid in BASELINE:
        b = BASELINE[cid]
        c = CURRENT.get(cid)
        flag = "REGRESSION" if cid in regs else "ok"
        shown = f"{c:.2f}" if c is not None else "MISSING"
        print(f"  {cid}: baseline={b:.2f} current={shown:>7}  {flag}")
    verdict = gate(regs)
    print(f"\nverdict: {verdict}")
    assert "c3" in regs, "c3 should be flagged"
    assert "c1" not in regs, "c1 (0.95->0.94) within tolerance"
    assert verdict == "FAIL", "critical regression must fail the gate"
    assert gate(detect_regressions(BASELINE, BASELINE, TOL)) == "PASS", "clean run passes"
    print("self-check OK: detects real regressions, ignores noise, gates correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
