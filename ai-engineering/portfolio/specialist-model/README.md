# The Specialist Model

The **control plane** for a LoRA/QLoRA fine-tune: dataset engineering, LoRA/QLoRA config
generation + validation, a **base-vs-tuned benchmark harness**, and W&B-style metric
logging. Week 7's weekly build from the SDE → AI Engineer course, packaged as a portfolio
service.

> ⚠️ The heavy PEFT/QLoRA training itself runs on **cloud GPUs**, never on the dev laptop
> (8 GB constraint). This repo is the fully-offline, testable layer that *drives and
> verifies* the fine-tune — dataset prep, config, and the benchmark that proves the
> specialist beats the base model.

## What it does
- **Dataset engineering**: formats raw examples into chat-style SFT records, dropping empties.
- **LoRA/QLoRA config**: generates + validates adapter configs (rank, alpha, dropout,
  target modules, 4/8-bit quantization).
- **Benchmark harness**: scores base vs tuned on a domain task and emits a verdict
  (`SPECIALIST_WINS` / `TIE` / `REGRESSION`) — the payload for a benchmark dashboard.
- **W&B-style logging**: in-memory metric history (swap in real `wandb` in prod).

## Quick start
```bash
cd ai-engineering/portfolio/specialist-model
pip install -e . pytest
pytest -q                 # offline, no model downloads, no GPU
uvicorn specialist_model.app:app --port 8000
curl -X POST localhost:8000/config -H 'content-type: application/json' -d '{"r":8,"lora_alpha":16,"quantization_bit":4}'
```

## Layout
```
specialist_model/
  core.py   # Example, build_sft_dataset, LoraConfig, validate_lora_config, run_benchmark, compare, WandbLogger
  app.py    # FastAPI wrapper (/health /ready /dataset /config /benchmark)
tests/test_core.py
docs/            # PLANNING, DESIGN (mermaid), DEPLOYMENT
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Production notes
Wire `build_sft_dataset` output into a PEFT/QLoRA trainer on cloud GPUs; log to W&B via the
real client; feed `compare()` results to a Streamlit dashboard. The scorer is pluggable —
swap the heuristic for a real eval (exact-match, LLM-judge, RAGAS) without touching the harness.
