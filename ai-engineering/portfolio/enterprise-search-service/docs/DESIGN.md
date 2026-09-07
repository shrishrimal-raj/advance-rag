# DESIGN — Enterprise Search Service

## 1. Pipeline overview

```mermaid
flowchart LR
    Q[Query] --> T[Tokenize]
    T --> B[BM25 scores]
    T --> D[Dense cosine]
    B --> BR[rank list]
    D --> DR[rank list]
    BR --> RRF[RRF fusion]
    DR --> RRF
    RRF --> TOP[top-k]
    TOP --> RR[Rerank]
    RR --> OUT[Results]
```

## 2. Retrieval + fusion sequence

```mermaid
sequenceDiagram
    participant C as Caller
    participant S as SearchService
    participant I as Index
    C->>S: search(query, top_k)
    S->>I: bm25(qtoks)
    I-->>S: {doc: score}
    S->>I: dense(qtoks)
    I-->>S: {doc: cosine}
    Note over S: sort each into rank lists
    S->>I: rrf([bm_rank, dense_rank])
    I-->>S: fused scores
    Note over S: take top-k, rerank by joint F1
    S-->>C: ordered results
```

## 3. RRF math
For each document, `RRF(d) = Σ over lists of 1/(k + rank_in_list + 1)`, with `k=60`.
Rank-based, so it needs no calibration between BM25 and cosine scales.

## 4. Data model

```mermaid
classDiagram
    class Index {
        docs[]
        tokens map
        df Counter
        avgdl float
        vocab[]
        dvec map
        bm25(qtoks) map
        dense(qtoks) map
        rrf(lists) map
        search(query, top_k) list
    }
    class SearchService {
        Index index
        search(query, top_k) list
    }
    SearchService --> Index
```

## 5. Failure / edge handling

```mermaid
flowchart TD
    A[query] --> B{tokenize non-empty?}
    B -->|no| C[return empty list]
    B -->|yes| D[score bm25 + dense]
    D --> E[fuse via RRF]
    E --> F[take top-k]
    F --> G{rerank enabled?}
    G -->|yes| H[joint F1 rerank + sort]
    G -->|no| I[return fused order]
    H --> J[results]
    I --> J
```

## Key decisions
- **Rank-based fusion (RRF)** — avoids fragile cross-method score normalization.
- **Deterministic dense proxy** — offline-testable; clean swap-in point for real embeddings.
- **Rerank only top-k** — keeps the expensive joint scoring cheap.
