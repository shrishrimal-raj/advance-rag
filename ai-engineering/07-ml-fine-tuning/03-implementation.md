# 🔧 Week 7 Implementation — Build The Domain-Specific Model

> Step-by-step. Each step ends with a **verify** line. ⚠️ Heavy week: run only the offline parts here.

## 0. Setup
```bash
./.venv/bin/python -c "import torch, transformers, datasets; print('ok')"
```
**verify:** prints `ok`.

## 1. Prove the training loop (approach_2)
A tiny 2-layer PyTorch MLP learns a synthetic regression/classification task. Log train+val loss per epoch; assert val loss drops (convergence) and show the overfitting guard.
**verify:** `python code/approach_2_raw_training_loop.py` converges (final val loss < initial) and passes its self-check.

## 2. Dataset prep (approach_3)
Turn raw `{prompt, completion}` examples into a training-ready dataset: apply a chat template, report token-length stats, and do a stratified train/val split with no leakage.
**verify:** `python code/approach_3_dataset_prep_from_scratch.py` produces correct split sizes and passes its self-check.

## 3. Fine-tune vs RAG (benchmark)
A deterministic decision matrix scoring both approaches across criteria (freshness, cost, latency, format control, data availability) and emitting a recommendation.
**verify:** `python code/benchmark_finetuned_vs_rag.py` recommends correctly for two contrasting scenarios and passes its self-check.

## 4. The full pipeline (main.py)
Complete HF+PEFT flow: load dataset -> tokenize -> LoRA config (peft if installed, else full FT) -> TrainingArguments -> train -> eval. **Dry-run by default** (prints every step's config, loads no model). `--run --model <name>` performs the real training (needs GPU + model download - not on this machine).
**verify:** `python code/main.py` (dry-run) prints the full plan and exits 0 without loading a model.

## 5. Run everything (offline subset)
```bash
PY=./.venv/bin/python   # or .venv\Scripts\python.exe
$PY code/approach_2_raw_training_loop.py
$PY code/approach_3_dataset_prep_from_scratch.py
$PY code/benchmark_finetuned_vs_rag.py
$PY code/main.py            # dry-run only
```

## Troubleshooting
- **`--run` OOMs** - expected on 8 GB; use a smaller model + LoRA + short max_length on real hardware.
- **peft missing** - main.py falls back to full fine-tune; install `peft` for adapters.
- **Val loss rising** - reduce epochs / add early stopping; you're overfitting.
- **Template mismatch** - confirm the chat template matches what the trainer tokenizes.
