# DESIGN — AI Gateway Service

## 1. Component overview
The gateway sits between callers and N ordered LLM providers. It owns retry,
fallback, rate limiting, and logging. Providers are opaque callables.

```mermaid
flowchart LR
    C[Caller] --> G[AIGateway]
    G --> RL{Rate limit?}
    RL -->|no tokens| D[503 degraded]
    RL -->|ok| P1[Provider 1]
    P1 -->|fail x retries| P2[Provider 2]
    P2 -->|fail| P3[Provider N]
    P1 -->|ok| R[Result]
    P2 -->|ok| R
    P3 -->|all fail| D
```

## 2. Request lifecycle

```mermaid
sequenceDiagram
    participant C as Caller
    participant G as Gateway
    participant B as TokenBucket
    participant P as Provider
    C->>G: complete(prompt)
    G->>B: allow()
    alt no tokens
        B-->>G: False
        G-->>C: 503 rate limited
    else ok
        loop each provider, up to max_retries
            G->>P: prompt
            alt success
                P-->>G: text
                G-->>C: 200 {text, provider}
            else error
                P-->>G: exception
                Note over G: log warn, try next attempt/provider
            end
        end
        G-->>C: 503 degraded (all failed)
    end
```

## 3. Retry + fallback state machine

```mermaid
stateDiagram-v2
    [*] --> RateCheck
    RateCheck --> Limited : no tokens
    RateCheck --> TryProvider : tokens available
    TryProvider --> Success : provider returns
    TryProvider --> NextAttempt : error and attempts < max_retries
    TryProvider --> NextProvider : retries exhausted for this provider
    NextAttempt --> TryProvider
    NextProvider --> TryProvider : more providers left
    NextProvider --> Degraded : no providers left
    Success --> [*]
    Limited --> [*]
    Degraded --> [*]
```

## 4. Data model

```mermaid
classDiagram
    class GatewayConfig {
        int max_retries
        float timeout_s
        int rate_limit_per_min
    }
    class TokenBucket {
        int capacity
        float tokens
        float refill_per_s
        bool allow()
    }
    class GatewayResult {
        bool ok
        str text
        str provider
        int attempts
        bool degraded
    }
    class AIGateway {
        providers[]
        GatewayConfig config
        TokenBucket bucket
        GatewayResult complete(prompt)
    }
    AIGateway --> GatewayConfig
    AIGateway --> TokenBucket
    AIGateway --> GatewayResult
```

## 5. Failure modes & recovery

```mermaid
flowchart TD
    A[Provider raises] --> B{attempts left?}
    B -->|yes| C[retry same provider]
    B -->|no| D{more providers?}
    D -->|yes| E[fallback to next]
    D -->|no| F[503 degraded + error log]
    C --> A
    E --> A
```

## Key decisions
- **Providers are callables** — no SDK coupling; trivially swappable/testable.
- **Token bucket** over fixed window — smooths bursts, standard technique.
- **Graceful degradation** — a downed gateway returns 503, never an unhandled crash.
