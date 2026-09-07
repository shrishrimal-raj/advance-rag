FROM python:3.12-slim

WORKDIR /srv
COPY . .
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "knowledge_engine.app:app", "--host", "0.0.0.0", "--port", "8000"]
