# DESIGN — The Dual-Agent Supervisor

## 1. Maker-checker loop

```mermaid
flowchart TD
    T[task] --> M[maker agent]
    M --> C{checker approves?}
    C -->|yes| OUT[return output]
    C -->|no| R{retries left?}
    R -->|yes| M
    R -->|no| FAIL[return failure]
```

## 2. Failover per agent

```mermaid
sequenceDiagram
    participant S as Supervisor
    participant P as Primary provider
    participant F as Fallback provider
    S->>P: call(task)
    alt primary healthy
        P-->>S: result (provider=primary)
    else primary outage
        P--xS: raise ProviderOutage
        S->>F: call(task)
        F-->>S: result (provider=fallback)
    end
```

## 3. Retry state machine

```mermaid
stateDiagram-v2
    [*] --> Attempt1
    Attempt1 --> Approved : checker ok
    Attempt1 --> Attempt2 : rejected & retries left
    Attempt2 --> Approved : checker ok
    Attempt2 --> Exhausted : no retries left
    Approved --> [*]
    Exhausted --> [*]
```

## 4. Data model

```mermaid
classDiagram
    class CheckerVerdict {
        bool approved
        str reason
    }
    class DualAgentSupervisor {
        callable maker
        callable checker
        int max_retries
        list trace
        dict run(task)
    }
    class ProviderOutage
    DualAgentSupervisor --> CheckerVerdict
```

## Key decisions
- **Failover is broad** — any exception on the primary triggers the fallback (outages are
  messy; don't guess which error means "down").
- **Bounded by design** — `max_retries` is a hard cap; the loop can never spin forever.
- **Trace everything** — every attempt records maker value, provider, and verdict for audit.
