# 🔧 Week 9 Implementation — Build The Document Intelligence Pipeline

> Step-by-step. Each step ends with a **verify** line. ⚠️ Heavy week: run only the offline parts + the 1-call selftest here.

## 0. Setup
```bash
./.venv/bin/python -c "import numpy; print('ok')"
```
**verify:** prints `ok`.

## 1. CLIP-style fusion (approach_2)
Hand-defined image/text vectors in a shared space; rank texts per image by cosine similarity. Assert each image's top match is its true caption.
**verify:** `python code/approach_2_clip_style_fusion_from_scratch.py` retrieves correctly and passes its self-check.

## 2. Audio front-end (approach_3)
Synthesize a waveform with tone bursts; compute an energy envelope; segment voiced regions by threshold (VAD). Assert the right number of segments and approximate durations.
**verify:** `python code/approach_3_audio_speech_from_scratch.py` segments correctly and passes its self-check.

## 3. Fusion strategy (benchmark)
Score unimodal vs early vs late fusion across accuracy/cost/latency/data-needs for three scenarios; recommend. Assert the right strategy wins each.
**verify:** `python code/benchmark_modality_fusion.py` recommends correctly and passes its self-check.

## 4. The pipeline (main.py)
Stages: extract (OCR/vision - simulated on a sample here) -> clean -> understand/structure (one cloud LLM call emits JSON fields). **Dry-run by default** (prints stages, no model). `--selftest` runs the full chain on a sample invoice with 1 LLM call. `--run --image PATH` would invoke a real OCR/vision model (not on this machine).
**verify:** `python code/main.py --selftest` extracts structured JSON from the sample (1 LLM call).

## 5. Run everything
```bash
PY=./.venv/bin/python   # or .venv\Scripts\python.exe
$PY code/approach_2_clip_style_fusion_from_scratch.py
$PY code/approach_3_audio_speech_from_scratch.py
$PY code/benchmark_modality_fusion.py
$PY code/main.py --selftest
```

## Troubleshooting
- **No OCR/vision model** - expected here; the extract stage is simulated, real OCR plugs in at that seam.
- **CLIP scores too close** - increase separation between matched/mismatched vectors; the contrastive gap is the point.
- **VAD over-segments** - raise the energy threshold or add a minimum segment length.
- **LLM returns non-JSON** - request strict JSON and parse defensively (regex the first `{...}`).
