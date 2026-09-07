# Vector Databases & Indexing Notes

## What is a Vector Database?
A vector database stores high-dimensional vectors (embeddings) and supports fast similarity search. Unlike SQL databases optimized for exact matches, vector DBs are built for approximate nearest neighbor (ANN) search over millions of points.

## Indexing Strategies

### Flat (Brute Force)
Computes distance to every vector. Exact results but O(n) per query. Fine below ~100k vectors.

### IVF (Inverted File Index)
Partitions space into Voronoi cells using k-means centroids. At query time, only the `nprobe` nearest cells are searched. Trade-off: more probes = more accurate but slower. Typical config: nlist = sqrt(N), nprobe = 10-100.

### HNSW (Hierarchical Navigable Small World)
Builds a multi-layer graph. Upper layers are sparse highways for long jumps; lower layers are dense for fine-grained navigation. Parameters:
- `M` (max edges per node): default 16, higher = better recall, more memory
- `efConstruction`: build-time beam width, higher = better index quality, slower build
- `efSearch`: query-time beam width, higher = better recall, slower queries

HNSW typically achieves >95% recall at sub-millisecond latency and is the default choice in ChromaDB, Qdrant, pgvector, Milvus.

## Distance Metrics
- **Cosine similarity**: direction only; standard for text embeddings normalized to unit length
- **L2 (Euclidean)**: magnitude matters; common for image embeddings
- **Inner product**: equivalent to cosine when vectors are L2-normalized

## Production Considerations
- Persist indexes to disk; rebuild on embedding model change (vectors are not portable across models!)
- Shard by tenant/metadata for multi-tenant SaaS
- Combine with scalar filters (pre-filter vs post-filter trade-off)
