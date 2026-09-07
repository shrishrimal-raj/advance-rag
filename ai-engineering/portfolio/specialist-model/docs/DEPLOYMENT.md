# DEPLOYMENT — The Specialist Model

## Local (offline control plane)
```bash
pip install -e .
uvicorn specialist_model.app:app --port 8000
curl -X POST localhost:8000/config -H 'content-type: application/json' -d '{"r":8,"lora_alpha":16,"quantization_bit":4}'
curl -X POST localhost:8000/dataset -H 'content-type: application/json' -d '{"examples":[{"instruction":"Summarize","input":"doc","output":"summary"}]}'
```

## Docker
```bash
docker build -t specialist-model .
docker run -p 8000:8000 -e WANDB_API_KEY=wandb-... -e BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct specialist-model
```

## Compose
```bash
cp .env.example .env   # fill keys
docker compose up -d
curl localhost:8000/health
curl localhost:8000/ready   # {"ready": true, "note": "control plane; heavy training runs on cloud GPUs"}
```

## CI
`.github/workflows/ci.yml` runs `pip install -e . pytest` then `pytest -q`. Tests are fully
offline — no model downloads, no GPU, no secrets required.

## Training rollout (cloud)
- [ ] Feed `build_sft_dataset` output to a PEFT/QLoRA trainer on cloud GPUs (`TRAINING_COMPUTE`).
- [ ] Log to W&B with the real client using `WANDB_API_KEY`.
- [ ] Run `compare()` on the domain eval set; require `SPECIALIST_WINS` before serving.
- [ ] Serve the tuned adapter; keep the base model as fallback.
