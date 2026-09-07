# PLANNING — The Specialist Model

## Problem
A fine-tuned "specialist" is only worth it if it measurably beats the base model on the
domain task. Building that requires disciplined dataset engineering, sane adapter configs,
and a reproducible benchmark — plus a way to prove the win.

## Goals
1. Dataset engineering: raw examples → clean chat-style SFT records.
2. LoRA/QLoRA config generation + validation (catch bad hyperparams early).
3. Benchmark harness: base vs tuned, per-metric + mean, clear verdict.
4. W&B-style metric logging.
5. Fully offline-testable (no model download, no GPU).

## Non-goals
- Actual PEFT/QLoRA training (runs on cloud GPUs; documented path, not executed here).
- Real embedding/eval models (pluggable scorer; heuristic stand-in for tests).
- Dashboard UI (the harness emits the dashboard payload).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | Example + build_sft_dataset | ✅ |
| M2 | LoraConfig + validate_lora_config | ✅ |
| M3 | run_benchmark + compare (verdict) | ✅ |
| M4 | WandbLogger (in-memory) | ✅ |
| M5 | FastAPI + Docker + CI | ✅ |

## Risks & mitigations
- Overfitting / bad config → validate_lora_config flags common mistakes before training.
- Unproven win → compare() gives an explicit SPECIALIST_WINS/REGRESSION verdict.
- Heavy compute → training is out of scope for the laptop by design.
