# Papeer — capstone RAG assistant (Streamlit UI)
# Build context: repository root (see ../docker-compose.yml)
#   docker build -f 10-capstone-project/code/deploy/Dockerfile -t papeer .
# Run:
#   docker run --rm -p 8501:8501 --env-file .env -v papeer-chroma:/app/data/chroma_db papeer
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache

# curl is needed by the HEALTHCHECK; uv manages the venv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app

# Install locked deps first for layer caching (source changes don't re-install deps).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the project (module code lives under 10-capstone-project/).
COPY . .

# Non-root user + writable data dir for the persistent vector store.
RUN useradd -m appuser \
    && mkdir -p /app/data/chroma_db \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "10-capstone-project/code/ui/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
