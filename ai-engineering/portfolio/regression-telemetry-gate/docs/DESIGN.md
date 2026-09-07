# DESIGN — Regression Telemetry Gate

## 1. Pipeline overview

```mermaid
flowchart LR
    I[Items] --> E[run_evals + scorer]
    E --> M[Metrics: mean, pass_rate]
    B[Baseline] --> C[compare]
    M --> C
    C --> V{passed?}
    V --> T[telemetry]
    T --> W[write file / return]
```

## 2. Gate decision flow

```mermaid
flowchart TD
    A[metrics + baseline] --> B{pass_rate >= min?}
    B -->|no| R1[reason: low pass rate]
    B -->|yes| C{baseline present?}
    C -->|no| P[passed]
    C -->|yes| D{delta_mean < -threshold?}
    D -->|yes| R2[reason: regression]
    D -->|no| P
    R1 --> F[failed]
    R2 --> F
```

## 3. Sequence

```mermaid
sequenceDiagram
    participant CI as CI / Caller
    participant G as Gate
    participant S as Scorer
    participant T as Telemetry
    CI->>G: run(items, baseline)
    loop each item
        G->>S: score(answer, expected)
        S-->>G: float
    end
    Note over G: aggregate metrics
    G->>G: compare vs baseline
    G->>T: emit_telemetry(verdict)
    G-->>CI: verdict (passed, reasons)
```

## 4. Data model

```mermaid
classDiagram
    class GateConfig {
        float pass_threshold
        float regression_threshold
        float min_pass_rate
    }
    class Metrics {
        int n
        float mean_score
        float pass_rate
    }
    class GateVerdict {
        bool passed
        list reasons
        Metrics current
        Metrics baseline
        float delta_mean
        dict telemetry
    }
    class Gate {
        GateConfig config
        scorer
        run(items, baseline) GateVerdict
    }
    Gate --> GateConfig
    Gate --> GateVerdict
```

## 5. Failure modes

```mermaid
flowchart TD
    A[eval run] --> B{any item errors?}
    B -->|scorer raises| C[score clamped / treated as 0]
    B -->|ok| D[aggregate]
    D --> E{regression?}
    E -->|yes| F[block build]
    E -->|no| G[allow build]
```

## Key decisions
- **Pluggable scorer** — gate logic independent of how quality is measured.
- **Rank-free thresholds** — simple, explainable pass/fail rules.
- **Telemetry always emitted** — even on pass, for trend tracking.
