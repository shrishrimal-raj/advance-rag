# PLANNING — The Multimodal RAG Engine

## Problem
Real documents mix text, screenshots, and voice notes. A text-only RAG pipeline misses the
evidence sitting in images and audio. A multimodal engine must route each modality to the
right processor and fuse the results.

## Goals
1. Modality routing: classify input → dispatch to the correct processor.
2. Evidence fusion: combine text/image/audio content into one ranked context.
3. Live dashboards: per-modality counts, errors, success rate.
4. Resilience: one bad file degrades gracefully, never crashes the query.
5. Fully offline-testable (pluggable processors; no CLIP/Whisper required).

## Non-goals
- Real CLIP/Whisper inference (pluggable stand-ins; swap-in point documented).
- Vector store persistence (fusion returns the context; store it downstream).
- Front-end dashboard UI (the `/dashboard` endpoint is the contract).

## Milestones
| # | Deliverable | Status |
|---|-------------|--------|
| M1 | detect_modality + Evidence | ✅ |
| M2 | MultimodalRAGEngine.process (routing) | ✅ |
| M3 | fuse (ranked multi-modality context) | ✅ |
| M4 | DashboardMetrics (live snapshot) | ✅ |
| M5 | FastAPI + Docker + CI | ✅ |

## Risks & mitigations
- Bad media file → processor exceptions caught, counted as error, query continues.
- Missing processor → unknown modality recorded as error, not a crash.
- Unproven fusion quality → pluggable processors let you A/B real models without code change.
