# DESIGN — The Multimodal RAG Engine

## 1. Routing + fusion

```mermaid
flowchart TD
    IN[input item] --> D{detect_modality}
    D -->|text| TP[text processor]
    D -->|image| IP[image processor]
    D -->|audio| AP[audio processor]
    TP --> EV[Evidence]
    IP --> EV
    AP --> EV
    EV --> F[fuse: rank by score]
    F --> CTX[retrieval context]
```

## 2. Per-item processing

```mermaid
sequenceDiagram
    participant E as Engine
    participant P as Processor
    participant M as Metrics
    E->>E: detect_modality(item)
    E->>P: process(item)
    alt success
        P-->>E: content
        E->>M: record(modality, ok=true)
    else decode error / missing proc
        E->>M: record(modality, ok=false)
        E-->>E: empty Evidence (skipped)
    end
```

## 3. Fusion ranking

```mermaid
flowchart LR
    E1[Evidence text] --> S[sort by score desc]
    E2[Evidence image] --> S
    E3[Evidence audio] --> S
    S --> J["[modality] content" joined]
```

## 4. Data model

```mermaid
classDiagram
    class Evidence {
        str modality
        str content
        float score
    }
    class DashboardMetrics {
        int processed
        map by_modality
        int errors
        void record(modality, ok)
        dict snapshot()
    }
    class MultimodalRAGEngine {
        map processors
        DashboardMetrics metrics
        Evidence process(item)
        str fuse(items)
        dict dashboard()
    }
    MultimodalRAGEngine --> Evidence
    MultimodalRAGEngine --> DashboardMetrics
```

## Key decisions
- **Pluggable processors** — swap heuristic stand-ins for CLIP/Whisper without touching routing/fusion.
- **Fail soft** — a bad file is an error metric, not an exception; the query still returns what it can.
- **Ranked fusion** — best-scored evidence first, so the LLM sees the strongest signal up top.
