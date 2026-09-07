# DESIGN — The Capstone

## 1. Multi-agent graph

```mermaid
flowchart LR
    Q[query] --> P[planner]
    P --> R[researcher/RAG]
    R --> W[writer]
    W --> C[critic]
    C --> OUT[final]
```

## 2. Shared state flow

```mermaid
sequenceDiagram
    participant G as AgentGraph
    participant S as State
    participant N as Node
    G->>S: new State(query)
    loop each node
        G->>N: start event
        N->>S: mutate (plan/context/draft/final)
        G->>G: done event
        G-->>UI: stream(event, state)
    end
    G-->>UI: complete
```

## 3. Streaming (SSE)

```mermaid
flowchart TD
    A[/ask/stream/] --> GEN[generator]
    GEN --> Y1[yield planner:start]
    GEN --> Y2[yield researcher:done]
    GEN --> Y3[yield ...]
    GEN --> YN[yield complete]
    Y1 --> SSE[text/event-stream]
    Y2 --> SSE
    Y3 --> SSE
    YN --> SSE
```

## 4. Data model

```mermaid
classDiagram
    class State {
        str query
        str plan
        str context
        str draft
        str final
        list events
    }
    class AgentGraph {
        list nodes
        State run(query)
        stream(query)
    }
    AgentGraph --> State
```

## Key decisions
- **Shared mutable State** — like LangGraph's channel; each node reads/writes what it needs.
- **Stream per node** — the UI gets progress without waiting for the whole run.
- **Pluggable everything** — LLMs, retriever, and tracing are all injected, never hardcoded.
