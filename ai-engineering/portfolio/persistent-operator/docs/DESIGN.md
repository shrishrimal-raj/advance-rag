# DESIGN — The Persistent Operator

## 1. Externalized state

```mermaid
flowchart LR
    subgraph Process
        OP[PersistentOperator]
    end
    subgraph Store[Redis / MemoryStore]
        K[(operator:s1)]
    end
    OP -->|load on init| K
    OP -->|save after each mutation| K
    K -.restart.-> OP2[new instance loads same key]
```

## 2. Turn sequence

```mermaid
sequenceDiagram
    participant C as Client
    participant O as Operator
    participant S as Store
    participant T as Tool
    C->>O: step(user_input, decide)
    O->>S: save(user message)
    O->>O: decide(state)
    alt action = tool
        O->>T: invoke(name, args)
        T-->>O: result
        O->>S: save(tool call + result)
    else action = answer
        O->>S: save(assistant answer)
    end
    O-->>C: outcome
```

## 3. Restart continuity

```mermaid
stateDiagram-v2
    [*] --> Running : instance A
    Running --> Crashed : process dies
    Crashed --> Restarted : instance B, same store+session
    Restarted --> Running : state loaded from store
    note right of Restarted
        messages + tool_calls
        restored verbatim
    end note
```

## 4. Data model

```mermaid
classDiagram
    class MemoryStore {
        get(key)
        set(key, value)
        delete(key)
    }
    class PersistentOperator {
        store
        str session_id
        map tools
        dict _state
        void add_message(role, content)
        str call_tool(name, args)
        dict step(input, decide)
        dict history()
    }
    class Message {
        str role
        str content
    }
    class ToolCall {
        str name
        dict args
        str result
    }
    PersistentOperator --> MemoryStore
    PersistentOperator --> Message
    PersistentOperator --> ToolCall
```

## 5. Session isolation

```mermaid
flowchart TD
    A[session a] --> KA[(operator:a)]
    B[session b] --> KB[(operator:b)]
    KA -.->|never read by| B
    KB -.->|never read by| A
```

## Key decisions
- **Write-through persistence** — save after every mutation; no lost work on crash.
- **Store abstraction** — 3-method interface; Redis drops in without touching agent logic.
- **Injected `decide`** — keeps the LLM/LangGraph loop out of the core (testable, swappable).
