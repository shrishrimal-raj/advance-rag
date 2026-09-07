# DESIGN — The Enterprise Search Engine

## 1. Multi-tenant architecture

```mermaid
flowchart TB
    subgraph Index
        N1[acme namespace]
        N2[globex namespace]
        N3[demo namespace]
    end
    R[Router /search] -->|X-Tenant-ID: acme| N1
    R -->|X-Tenant-ID: globex| N2
    R -->|X-Tenant-ID: demo| N3
```

## 2. Request flow

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant E as Engine
    C->>A: POST /search {query} + X-Tenant-ID
    A->>E: search(tenant_id, query)
    E->>E: load ONLY that tenant's namespace
    E->>E: hybrid BM25+dense -> RRF -> top_k
    E-->>A: hits + citations
    A-->>C: results (scoped to tenant)
```

## 3. Hybrid fusion

```mermaid
flowchart TD
    Q[query] --> S[BM25 sparse ranks]
    Q --> D[dense cosine ranks]
    S --> F[RRF: sum 1/(k+rank)]
    D --> F
    F --> T[top_k + citations]
```

## 4. Data model

```mermaid
classDiagram
    class Doc {
        str id
        str text
        str source
        dict metadata
        list vector
    }
    class EnterpriseSearchEngine {
        int rrf_k
        map _namespaces
        void upsert(tenant, docs)
        list tenants()
        list search(tenant, query, vector, top_k, filters)
    }
    class Citation {
        str id
        str source
        int position
    }
    EnterpriseSearchEngine --> Doc
    EnterpriseSearchEngine --> Citation
```

## 5. Isolation guarantee

```mermaid
stateDiagram-v2
    [*] --> ResolveTenant
    ResolveTenant --> LoadNamespace : X-Tenant-ID
    LoadNamespace --> HybridRank : only this tenant's docs
    HybridRank --> Return : top_k
    Return --> [*]
    note right of LoadNamespace
        Other tenants' docs are
        never loaded into scope
    end note
```

## Key decisions
- **Namespace-per-tenant** — hard isolation at the data layer, not a post-filter.
- **Hybrid + RRF** — robust recall without score calibration.
- **Tenant via header** — clean trust boundary for a fronting auth gateway.
