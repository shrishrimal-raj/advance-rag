# 🎯 Week 07 — ML & Fine-Tuning

> Status: ✅ BUILT (see `../CHECKPOINT.md`).

## Objective
Adapt pre-trained models to specific domains & tasks.

## What You'll Learn
- Fine-Tuning vs RAG Trade-offs
- Parameter-Efficient Fine-Tuning (PEFT/LoRA)
- Dataset Preparation & Training Loops

## Tools / Stack
- **HuggingFace Transformers** (installed) · **datasets** (installed) · **PyTorch** (installed) · **PEFT** (documented; not preinstalled here)

> ⚠️ **Heavy workload.** Real fine-tuning needs a GPU + a multi-hundred-MB model download, which the 8 GB / no-big-download rule forbids. So this week is **code + docs**: `main.py` is a complete, guarded HF+PEFT pipeline that **dry-runs by default** (no model load) and only trains under `--run`. The *mechanics* are proven by a **runnable PyTorch toy training loop** (`approach_2`) on synthetic data - tiny and fully offline.

## Weekly Build
**The Domain-Specific Model** — A fine-tuned model for a specific task (complete pipeline + runnable training-loop demo).

**Outcome:** Understand exactly what fine-tuning costs and how to run it when you have the hardware.

## Approach details
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | Full HF+PEFT fine-tune pipeline (dry-run default; `--run` trains) | dry-run **offline**; `--run` needs GPU+model |
| `code/approach_2_raw_training_loop.py` | PyTorch toy training loop (forward/loss/backward/update) | **fully offline** (tiny MLP, synthetic data) |
| `code/approach_3_dataset_prep_from_scratch.py` | Build + split an instruction dataset from raw examples | **fully offline** |
| `code/benchmark_finetuned_vs_rag.py` | Fine-tune vs RAG decision matrix | **fully offline** |

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` · `approach_2_raw_training_loop.py` · `approach_3_dataset_prep_from_scratch.py` · `benchmark_finetuned_vs_rag.py`
- [x] All scripts: UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four.
- Run `main.py` (dry-run, no model load), `approach_2` (toy loop converges), `approach_3`, `benchmark` - all offline.
- Do NOT run `main.py --run` here (heavy).

## Prerequisites
- Weeks 1–6 complete. `torch`/`transformers`/`datasets` installed. No Ollama; no >100 MB downloads.

## Time Estimate
~5–7 hours hands-on (plus real training time on suitable hardware).
