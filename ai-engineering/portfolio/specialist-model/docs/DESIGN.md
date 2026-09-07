# DESIGN — The Specialist Model

## 1. Control-plane overview

```mermaid
flowchart LR
    RAW[raw examples] --> DS[build_sft_dataset]
    CFG[LoraConfig] --> VAL[validate_lora_config]
    DS --> TRAIN[cloud QLoRA trainer]
    VAL --> TRAIN
    TRAIN --> BASE[base model scores]
    TRAIN --> TUNED[tuned model scores]
    BASE --> CMP[compare]
    TUNED --> CMP
    CMP --> DASH[benchmark dashboard]
    TRAIN --> WB[W&B logger]
```

## 2. Dataset engineering

```mermaid
flowchart TD
    E[Example] --> C{instruction non-empty AND output non-empty?}
    C -->|no| DROP[drop]
    C -->|yes| F[format chat messages user+assistant]
    F --> OUT[SFT record]
```

## 3. Config validation

```mermaid
stateDiagram-v2
    [*] --> Check
    Check --> Valid : all rules pass
    Check --> Invalid : any rule fails
    Valid --> [*]
    Invalid --> [*]
    note right of Check
        r>0, alpha>=r,
        dropout in [0,1),
        target_modules non-empty,
        quant in {None,4,8}
    end note
```

## 4. Benchmark + verdict

```mermaid
sequenceDiagram
    participant H as Harness
    participant S as Scorer
    participant B as Base
    participant T as Tuned
    H->>S: score(base cases)
    S-->>H: base scores + mean
    H->>S: score(tuned cases)
    S-->>H: tuned scores + mean
    H->>H: delta = tuned.mean - base.mean
    alt delta > 0
        H-->>H: SPECIALIST_WINS
    else delta == 0
        H-->>H: TIE
    else delta < 0
        H-->>H: REGRESSION
    end
```

## 5. Data model

```mermaid
classDiagram
    class Example {
        str instruction
        str input
        str output
    }
    class LoraConfig {
        int r
        int lora_alpha
        float lora_dropout
        list target_modules
        int quantization_bit
        bool is_qlora
        dict to_dict()
    }
    class BenchmarkResult {
        str model
        dict scores
        float mean
    }
    class WandbLogger {
        str run_name
        list history
        void log(step, **metrics)
        dict summary()
    }
    BenchmarkResult --> Example
```

## Key decisions
- **Control plane only** — the laptop never trains; it prepares, validates, and proves.
- **Explicit verdict** — no ambiguous "looks better"; compare() returns a machine-readable result.
- **Pluggable scorer** — swap heuristic for exact-match / LLM-judge / RAGAS without changing the harness.
