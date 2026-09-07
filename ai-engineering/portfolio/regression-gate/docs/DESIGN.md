# DESIGN — The Regression Gate

## 1. Gate pipeline

```mermaid
flowchart LR
    G[Golden dataset] --> S[score_dataset]
    S --> C[current metrics]
    B[stored baseline] --> CMP[compare]
    C --> CMP
    CMP --> V{any drop > threshold?}
    V -->|no| P[PASS]
    V -->|yes| F[FAIL + regressions]
```

## 2. CI integration

```mermaid
sequenceDiagram
    participant Dev
    participant PR as Pull Request
    participant CI as GitHub Actions
    participant Gate as Regression Gate
    Dev->>PR: push change
    PR->>CI: trigger workflow
    CI->>Gate: POST /gate {cases, baseline}
    Gate-->>CI: {passed, regressions}
    alt passed
        CI-->>PR: allow merge
    else failed
        CI-->>PR: block merge (list regressions)
    end
```

## 3. Metric computation

```mermaid
flowchart TD
    A[case: q,a,c,e] --> M1[answer_correctness = overlap(a,e)/e]
    A --> M2[context_relevance = overlap(c,q)/q]
    A --> M3[faithfulness = overlap(a,c)/a]
    M1 --> AVG[mean across cases]
    M2 --> AVG
    M3 --> AVG
```

## 4. Data model

```mermaid
classDiagram
    class EvalCase {
        str question
        str answer
        str context
        str expected
    }
    class GateResult {
        bool passed
        dict current
        dict baseline
        list regressions
        dict as_dict()
    }
    class METRICS {
        <<map>>
        name -> fn(q,a,c,e)
    }
    GateResult --> EvalCase
```

## 5. Verdict state machine

```mermaid
stateDiagram-v2
    [*] --> Scoring
    Scoring --> Comparing : current computed
    Comparing --> Pass : all deltas >= -threshold
    Comparing --> Fail : any delta < -threshold
    Pass --> [*]
    Fail --> [*]
```

## Key decisions
- **Deterministic metrics** — reproducible in CI; no flaky LLM calls on every PR.
- **Threshold-based verdict** — simple, tunable, explainable (lists exact regressions).
- **Swappable scorers** — metric functions are the only thing to replace for RAGAS/LangSmith.
