# DESIGN — The Audited Tool Bridge

## 1. Gateway pipeline

```mermaid
flowchart LR
    C[Agent] --> K{valid API key?}
    K -->|no| D1[deny: unauthorized]
    K -->|yes| R{rate limit OK?}
    R -->|no| D2[deny: rate_limited]
    R -->|yes| T{tool exists?}
    T -->|no| D3[deny: unknown_tool]
    T -->|yes| X[dispatch tool]
    X --> A[audit log]
    D1 --> A
    D2 --> A
    D3 --> A
```

## 2. Call sequence

```mermaid
sequenceDiagram
    participant A as Agent
    participant B as ToolBridge
    participant T as Tool
    A->>B: invoke(api_key, tool, args)
    B->>B: lookup client by key
    alt bad key
        B-->>A: {ok:false, unauthorized}
    else ok
        B->>B: token bucket allow?
        B->>T: fn(**args)
        T-->>B: result
        B-->>A: {ok:true, result}
    end
    B->>B: append audit entry
```

## 3. Token bucket

```mermaid
stateDiagram-v2
    [*] --> Full : capacity tokens
    Full --> Draining : allow() consumes 1
    Draining --> Full : refill at rate/sec (capped)
    Draining --> Empty : tokens < 1
    Empty --> Deny : allow() returns False
    Empty --> Draining : refill crosses 1
```

## 4. Data model

```mermaid
classDiagram
    class TokenBucket {
        int capacity
        float refill
        bool allow()
    }
    class AuditEntry {
        float ts
        str client
        str tool
        bool allowed
        str reason
        dict args
        dict as_dict()
    }
    class ToolBridge {
        map valid_keys
        map tools
        list audit
        void register_tool(name, fn)
        dict invoke(api_key, tool, args)
        list audit_log(client)
    }
    ToolBridge --> TokenBucket
    ToolBridge --> AuditEntry
```

## 5. Audit flow

```mermaid
flowchart TD
    E[event] --> L[append AuditEntry]
    L --> Q{query?}
    Q -->|all| ALL[full log]
    Q -->|client| F[filter by client]
```

## Key decisions
- **Auth first** — nothing runs before the key check.
- **Audit every path** — allow, deny, and error all logged with a reason.
- **Per-client buckets** — one noisy client can't starve others.
