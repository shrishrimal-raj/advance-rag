import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 3 - Reciprocal Rank Fusion from scratch (pure python, no deps).

RRF merges two or more ranked lists into one without comparing raw scores:
    score(d) = sum over each list of  1 / (k + rank(d))     (rank is 1-based)
Runs fully offline; includes a self-check.
"""


def rrf(rank_lists, k=60):
    """rank_lists: list of lists of item ids (best first). Returns {id: score}."""
    scores = {}
    for ranks in rank_lists:
        for rank, item in enumerate(ranks, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_topk(rank_lists, k_final=5, k=60):
    scores = rrf(rank_lists, k=k)
    order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(item, round(score, 6)) for item, score in order[:k_final]]


def main():
    print("=== Approach 3: RRF from scratch (offline) ===")
    bm25 = ["d1", "d2", "d3", "d4", "d5"]
    dense = ["d2", "d1", "d5", "d3", "d4"]
    fused = rrf_topk([bm25, dense], k_final=5)
    for i, (doc, s) in enumerate(fused, 1):
        print(f"  #{i} {doc}  rrf={s}")
    order = [d for d, _ in fused]
    assert order.index("d2") < order.index("d4"), f"unexpected order: {order}"
    assert order[0] in ("d1", "d2"), f"top should be d1 or d2, got {order[0]}"
    print("self-check OK: fusion favors docs ranked high across lists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
