import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import statistics

"""Approach 3 - monitoring & maintenance (offline).

Simulate a batch of runs; compute p50/p95 latency, error rate, and a drift proxy
(mean answer length). Alert when a metric breaches a threshold. Detect an injected
anomaly (a slow/erroring window) plus answer-length drift.
"""


def percentile(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def summarize(runs):
    lats = [r["latency_ms"] for r in runs]
    errs = sum(1 for r in runs if r["error"])
    lens = [r["answer_len"] for r in runs]
    return {
        "n": len(runs),
        "p50_ms": round(percentile(lats, 0.5), 1),
        "p95_ms": round(percentile(lats, 0.95), 1),
        "error_rate": round(errs / len(runs), 3) if runs else 0.0,
        "mean_answer_len": round(statistics.mean(lens), 1) if lens else 0.0,
    }


def drift_alert(baseline_len, current_len, rel_thresh=0.5):
    if baseline_len <= 0:
        return False
    return abs(current_len - baseline_len) / baseline_len > rel_thresh


def check(summary, p95_limit=500, err_limit=0.05):
    alerts = []
    if summary["p95_ms"] > p95_limit:
        alerts.append(f"p95 latency {summary['p95_ms']}ms > {p95_limit}ms")
    if summary["error_rate"] > err_limit:
        alerts.append(f"error rate {summary['error_rate']} > {err_limit}")
    return alerts


def make_runs(n=100, anomaly_from=10 ** 9):
    runs = []
    for i in range(n):
        anomalous = i >= anomaly_from
        runs.append({
            "latency_ms": 900 if anomalous else 120 + (i % 5) * 10,
            "error": bool(anomalous and i % 2 == 0),
            "answer_len": 40 if anomalous else 200 + (i % 3) * 10,
        })
    return runs


def main():
    print("=== Approach 3: monitoring & maintenance (offline) ===")
    healthy = make_runs(100)
    hs = summarize(healthy)
    print("healthy:", hs)
    print("healthy alerts:", check(hs) or "none")
    mixed = make_runs(100, anomaly_from=70)
    ms = summarize(mixed)
    print("with-anomaly:", ms)
    alerts = check(ms)
    print("anomaly alerts:", alerts)
    first = summarize(mixed[:70])["mean_answer_len"]
    second = summarize(mixed[70:])["mean_answer_len"]
    print(f"drift check: first-half len={first}  second-half len={second}")
    assert check(hs) == [], "healthy batch must not alert"
    assert any("p95" in a for a in alerts), "must detect latency anomaly"
    assert any("error" in a for a in alerts), "must detect error-rate anomaly"
    assert drift_alert(first, second), "must detect answer-length drift"
    print("self-check OK: metrics computed, anomaly + drift detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
