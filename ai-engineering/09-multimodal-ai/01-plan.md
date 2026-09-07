# 🎯 Week 09 — Multimodal AI

> Status: ✅ BUILT (see `../CHECKPOINT.md`).

## Objective
Build systems that process multiple data types.

## What You'll Learn
- Vision Transformers & CLIP
- Audio Processing & Speech-to-Text
- Multimodal Fusion Strategies

## Tools / Stack
- **HuggingFace Transformers** (installed) · **OpenAI Whisper** (documented) · **NumPy** (installed) · **shared.config.get_llm** (cloud)

> ⚠️ **Heavy workload.** Real vision/audio models are large downloads - forbidden here. This week is **code + docs**: `main.py` is a complete, guarded document-intelligence pipeline (dry-run default; the *understand/structure* stage runs a single cloud LLM call on simulated OCR output). The multimodal *mechanics* are proven offline with NumPy: a CLIP-style shared-space matcher (`approach_2`) and a speech front-end (`approach_3`).

## Weekly Build
**The Document Intelligence Pipeline** — Extract, understand, and process documents with images.

**Outcome:** Master the architecture behind modern multimodal products.

## Approach details
| File | Approach | Runs |
|------|----------|------|
| `code/main.py` | Document-intelligence pipeline: extract -> clean -> understand/structure | dry-run **offline**; `--selftest` = 1 LLM call on simulated OCR |
| `code/approach_2_clip_style_fusion_from_scratch.py` | CLIP-style shared embedding space + cosine retrieval (NumPy) | **fully offline** |
| `code/approach_3_audio_speech_from_scratch.py` | Speech front-end: energy envelope + VAD segmentation (NumPy) | **fully offline** |
| `code/benchmark_modality_fusion.py` | Unimodal vs early vs late fusion decision matrix | **fully offline** |

## Deliverables (this week)
- [x] `02-learning.md` — theory + noob→expert mermaid diagrams (≥5)
- [x] `03-implementation.md` — step-by-step build guide
- [x] `code/main.py` · `approach_2_clip_style_fusion_from_scratch.py` · `approach_3_audio_speech_from_scratch.py` · `benchmark_modality_fusion.py`
- [x] All scripts: UTF-8 guard + graceful degradation + `py_compile` clean

## Verification
- `py_compile` all four.
- Run `approach_2` + `approach_3` + `benchmark` (offline, self-check asserts).
- Run `main.py --selftest` (1 cloud LLM call on simulated OCR output).

## Prerequisites
- Weeks 1–8 complete. `numpy` installed. No Ollama; no >100 MB downloads.

## Time Estimate
~5–7 hours hands-on.
