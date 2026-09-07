# RAG Evaluation & MLOps Harness — batch job image.
# No ports are exposed: this is a CLI, not a service.
# Build:  docker build -t rag-evaluation-harness .
# Run:    docker run --rm -v ${PWD}/out:/app/out rag-evaluation-harness
#         (or: docker compose run harness)

FROM python:3.11-slim

# Install uv (pinned tool, no pip dependency drift)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project metadata first for layer caching
COPY pyproject.toml uv.lock README.md ./
COPY harness ./harness
COPY cli.py ./cli.py
COPY evals ./evals
COPY tests ./tests

# Install locked dependencies (no dev extras needed)
RUN uv sync --frozen --no-dev

# Artifacts land in /app/out; mount a volume there to persist them.
RUN mkdir -p /app/out

# Entrypoint: run the offline evaluation (pipeline A — no API key required).
# Override with e.g. "python cli.py gate --baseline evals/baseline.json" if desired.
ENTRYPOINT ["uv", "run", "python", "cli.py", "run", "--pipeline", "A", "--dataset", "evals/dataset.jsonl", "--report", "out/report.md"]
