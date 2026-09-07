# 🔧 Week 5 Implementation — Build The AI Regression Gate

> Step-by-step. Each step ends with a **verify** line.

## 0. Setup
```bash
./.venv/bin/python -c "import ragas; print(ragas.__version__)"
```
**verify:** prints a version (0.4.x).

## 1. Define a labeled dataset
A few test cases: `{id, query, expected_answer, gold_facts[], retrieved_contexts[]}`. Gold facts are the atomic truths the answer must contain.
**verify:** each case has non-empty `gold_facts`.

## 2. Offline metric proxies (approach_3)
Implement from scratch:
- `context_recall` = fraction of `gold_facts` found in `retrieved_contexts`.
- `answer_correctness` = token overlap between answer and expected_answer.
- `faithfulness_proxy` = fraction of answer content-ngrams supported by context.
**verify:** a perfect case scores 1.0; a hallucinated answer scores low faithfulness.

## 3. Tracing (main.py)
Wrap each case in a trace. `try: import langfuse` -> use it; else append JSON lines to `traces.jsonl`. Same shape either way.
**verify:** running main writes a trace record per case (file or langfuse).

## 4. Regression detection (benchmark)
Given `baseline` and `current` score maps, flag ids where `current < baseline - tol`.
**verify:** a deliberately-dropped case is flagged; stable cases are not.

## 5. Wire the gate (main.py)
Compute current scores, diff vs embedded baseline, print a table + PASS/FAIL verdict.
**verify:** `python code/main.py` prints a verdict and passes its self-check.

## 6. Real RAGAS (approach_2)
Build `SingleTurnSample`s, list metrics from `ragas.metrics.collections`. `--dry-run` prints the plan (offline); `--run` calls `evaluate` (needs an LLM configured for RAGAS).
**verify:** `python code/approach_2_ragas.py --dry-run` constructs samples without error.

## 7. Run everything
```bash
PY=./.venv/bin/python   # or .venv\Scripts\python.exe
$PY code/approach_3_custom_metrics_from_scratch.py
$PY code/benchmark_regression_detector.py
$PY code/main.py
$PY code/approach_2_ragas.py --dry-run
```

## Troubleshooting
- **RAGAS LLM config** — `--run` needs RAGAS pointed at an LLM; use `--dry-run` to stay offline.
- **Metric import warning** — import from `ragas.metrics.collections`, not `ragas.metrics`.
- **Gate always fails** — check the tolerance and that baseline/current share the same case ids.
- **JSONL grows** — it's a demo artifact; rotate or delete between runs.
