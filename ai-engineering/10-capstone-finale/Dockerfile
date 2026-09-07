FROM python:3.12-slim

# Preserve the repo layout so `parents[2]` in code/app.py resolves to ai-engineering/
WORKDIR /srv
COPY ai-engineering/pyproject.toml ./ai-engineering/
COPY ai-engineering/shared ./ai-engineering/shared
COPY ai-engineering/10-capstone-finale ./ai-engineering/10-capstone-finale

RUN pip install --no-cache-dir langchain langchain-openai langgraph fastapi uvicorn numpy

ENV PYTHONUNBUFFERED=1
WORKDIR /srv/ai-engineering/10-capstone-finale/code
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
