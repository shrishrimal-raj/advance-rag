# Deployment — RAG Evaluation Harness

The harness runs in CI, not as a service:

```yaml
# .github/workflows/eval.yml (sketch)
- run: uv sync
- run: uv run pytest -v
- run: uv run python cli.py run --pipeline A --report out/report.md
- run: uv run python cli.py gate --baseline evals/baseline.json --tolerance 0.05
```

Gate exit 1 fails the pipeline. Update `evals/baseline.json` deliberately after accepted quality changes.