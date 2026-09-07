# DESIGN — The Knowledge Engine

## 1. Pipeline overview

```mermaid
flowchart LR
    Q[Query] --> F[Metadata Filter]
    F --> S[Sparse BM25]
    F --> D[Dense Cosine]
    S --> R[RRF Fusion]
    D --> R
    R --> C[Citations]
    C --> A[Answer Context]
    DB[(pgvector)] --> F
```

## 2. Retrieval sequence

```mermaid
sequenceDiagram
    participant C as Client
    participant E as KnowledgeEngine
    participant B as BM25
    participant V as Dense
    C->>E: retrieve(query, vector, filters)
    E->>E: filter chunks by metadata
    E->>B: scores(query)
    B-->>E: sparse ranks
    E->>V: cosine(query_vec, each)
    V-->>E: dense ranks
    E->>E: RRF fuse -> top_k
    E-->>C: hits + citations
```

## 3. RRF fusion

```mermaid
flowchart TD
    A[sparse ranks] --> K[1/(k+rank)]
    B[dense ranks] --> K
    K --> S[sum per chunk]
    S --> T[sort desc, take top_k]
```

## 4. Data model

```mermaid
classDiagram
    class Chunk {
        str id
        str text
        str source
        dict metadata
        list vector
    }
    class KnowledgeEngine {
        int rrf_k
        list chunks
        void index(chunks)
        list retrieve(query, vector, top_k, filters)
        dict answer_context(query, ...)
    }
    class Citation {
        str id
        str source
        int position
    }
    KnowledgeEngine --> Chunk
    KnowledgeEngine --> Citation
```

## 5. Metadata filtering

```mermaid
stateDiagram-v2
    [*] --> AllChunks
    AllChunks --> Filtered : apply metadata == value
    Filtered --> Ranked : hybrid score
    Ranked --> TopK : slice
    TopK --> [*]
```

## Key decisions
- **Hybrid + RRF** — robust to both exact-term and semantic queries; RRF needs no score calibration.
- **Filter-before-rank** — cheap scoping; avoids ranking out-of-scope docs.
- **Citations as first-class output** — traceability is a production requirement, not an afterthought.
