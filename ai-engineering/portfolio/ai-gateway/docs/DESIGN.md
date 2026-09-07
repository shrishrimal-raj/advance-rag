# DESIGN — The AI Gateway

## 1. Component overview
The gateway streams from a pluggable async provider, emits SSE events, and records
token/cost into a shared tracker.

```mermaid
flowchart LR
    C[Client] --> G[FastAPI /chat/stream]
    G --> SC[stream_chat]
    SC --> P[Async Provider]
    P -->|chunk| SC
    SC -->|SSE token| C
    SC --> T[TokenCostTracker]
    SC -->|SSE usage| C
    U[/GET /usage/] --> T
```

## 2. Streaming sequence

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant S as stream_chat
    participant P as Provider
    C->>A: POST /chat/stream {prompt, model}
    A->>S: stream_chat(prompt, model, provider)
    loop each chunk
        S->>P: next()
        P-->>S: chunk
        S-->>C: data: {"type":"token",...}
    end
    Note over S: compute tokens + cost + latency
    S->>S: tracker.record(usage)
    S-->>C: data: {"type":"usage",...}
    S-->>C: data: [DONE]
```

## 3. Request state machine

```mermaid
stateDiagram-v2
    [*] --> Start
    Start --> Streaming : first chunk
    Streaming --> Streaming : more chunks
    Streaming --> Accounting : provider exhausted
    Accounting --> Done : usage event emitted
    Done --> [*]
```

## 4. Data model

```mermaid
classDiagram
    class Usage {
        str model
        int input_tokens
        int output_tokens
        float cost_usd
        float latency_ms
        dict as_dict()
    }
    class TokenCostTracker {
        int requests
        int input_tokens
        int output_tokens
        float cost_usd
        void record(Usage)
        dict summary()
    }
    class MODEL_PRICING {
        <<map>>
        model -> {input, output}
    }
    TokenCostTracker --> Usage
```

## 5. Cost computation

```mermaid
flowchart TD
    A[prompt, completion] --> B[estimate_tokens ~4 chars/token]
    B --> C{model in price table?}
    C -->|yes| D[use model prices]
    C -->|no| E[use default prices]
    D --> F[cost = (in*pin + out*pout)/1e6]
    E --> F
    F --> G[round to 8 dp]
```

## Key decisions
- **Async generator + SSE** — non-blocking streaming, standard for LLM chat backends.
- **Pluggable async provider** — offline-testable; production wraps OpenAI/OpenRouter.
- **Centralized tracker** — one place for real-time spend accounting.
