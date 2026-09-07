"""
Module 4 - Vector Stores: Approach C - HNSW parameter tuning (from-scratch benchmark).

Run from the project root:
    uv run python 04-vector-stores/code/approach_3_hnsw_tuning.py

What this proves:
    HNSW is an approximate index with REAL knobs. This script builds ONE fixed
    ~500-doc synthetic corpus, indexes it under several (M, efConstruction)
    settings, queries each index at several query-time `ef` values, and scores
    every combination against a NUMPY BRUTE-FORCE ground truth:

        recall@k = |ANN top-k  ∩  exact top-k| / k   (averaged over queries)

    Then it prints a rich table of config vs insert-time / query-time / recall
    and a data-driven recommendation.

The knobs (ChromaDB collection metadata):
    hnsw:M                max edges per node in the graph (connectivity)
    hnsw:construction_ef  beam width while INSERTING nodes (build quality)
    hnsw:search_ef        beam width while QUERYING (speed/quality trade-off)

Ground truth: pure numpy cosine similarity over all 500 vectors - O(N) per
query, exact. That is what ANN is approximating.

NO LLM / NO API keys - local MiniLM embeddings (already cached) + ChromaDB.
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pathlib
import random
import shutil
import time

# Make the project root importable so we can reuse shared.config from any folder.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from shared.config import get_embeddings

console = Console()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMP_DIR = PROJECT_ROOT / "data" / "chroma_hnsw_tuning"   # temp persist dir under data/

N_DOCS = 500          # corpus size
K = 10                # recall@k
N_QUERIES = 20        # held-out queries for timing + recall
SEED = 42

# (M, efConstruction) build settings to compare.
BUILD_CONFIGS = [
    (4, 16),
    (8, 32),
    (16, 64),
    (32, 128),
]
# Query-time beam widths to sweep per build config.
QUERY_EFS = [16, 64, 128]


def make_corpus(n: int, seed: int):
    """Synthetic but TOPIC-COHERENT corpus: 10 topics x template sentences.

    Topic coherence matters: random word soup makes every vector look alike and
    recall becomes meaningless. Real-ish clusters give ANN something to find.
    """
    rng = random.Random(seed)
    topics = {
        "vector search": ["approximate nearest neighbor", "cosine similarity", "embedding space", "top-k retrieval"],
        "databases": ["transaction log", "index page", "write ahead", "b-tree node"],
        "graphs": ["shortest path", "graph traversal", "edge weight", "node degree"],
        "ml models": ["gradient descent", "neural network layer", "training epoch", "loss function"],
        "networking": ["packet routing", "latency budget", "tcp handshake", "dns lookup"],
        "security": ["access token", "encryption key", "audit log", "permission check"],
        "storage": ["disk block", "cache eviction", "compression ratio", "replica set"],
        "compilers": ["parse tree", "symbol table", "register allocation", "bytecode emit"],
        "databases": ["transaction log", "index page", "write ahead", "b-tree node"],
        "search": ["inverted index", "term frequency", "document ranking", "query expansion"],
    }
    names = list(topics.keys())
    docs = []
    for i in range(n):
        topic = names[i % len(names)]
        words = rng.sample(topics[topic], k=4)
        filler = rng.randint(3, 9)
        doc = f"Document {i} about {topic}: " + " ".join(words) + f" section {rng.randint(1, 50)}."
        # small per-doc noise so vectors are not identical within a topic
        doc += " " + " ".join(f"w{j}" for j in range(filler))
        docs.append(doc)
    return docs


def make_queries(n: int, seed: int):
    """Held-out queries phrased like real user questions about the topics."""
    rng = random.Random(seed + 1)
    topics = ["vector search", "databases", "graphs", "ml models", "networking",
              "security", "storage", "compilers", "search"]
    qs = []
    for i in range(n):
        t = rng.choice(topics)
        qs.append(f"How does {t} work? Explain the core idea of {t}.")
    return qs


def brute_force_topk(qv: np.ndarray, mat: np.ndarray, k: int) -> list[int]:
    """Exact top-k by cosine similarity (numpy). qv: (D,), mat: (N, D) normalized."""
    sims = mat @ qv                      # cosine since rows are L2-normalized
    return list(np.argsort(-sims)[:k])


def main() -> None:
    console.rule("[bold cyan]Module 4 - Approach C: HNSW parameter tuning[/]")

    # ---- Setup -------------------------------------------------------------
    try:
        emb = get_embeddings()
    except Exception as exc:
        console.print(Panel(
            f"[red]Embedding model failed to load:[/]\n{exc}\n\n"
            "[bold]Hint:[/bold] needs the local MiniLM model (caches after first download).",
            title="Setup issue", border_style="red"))
        sys.exit(0)

    import chromadb

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    client = chromadb.PersistentClient(path=str(TEMP_DIR))
    console.print(f"\n[bold]Temp persist dir[/bold] -> {TEMP_DIR}")

    # ---- Build corpus + embed ONCE (reused across all configs) --------------
    console.print(f"\n[bold]Corpus[/bold]: generating {N_DOCS} synthetic docs + {N_QUERIES} queries...")
    docs = make_corpus(N_DOCS, SEED)
    queries = make_queries(N_QUERIES, SEED)
    t0 = time.perf_counter()
    doc_vecs = np.asarray(emb.embed_documents(docs), dtype=np.float32)
    qry_vecs = np.asarray(emb.embed_documents(queries), dtype=np.float32)
    console.print(f"  embedded {doc_vecs.shape[0]} docs + {qry_vecs.shape[0]} queries "
                  f"in {time.perf_counter() - t0:.1f}s  (dim={doc_vecs.shape[1]})")

    # Normalize once -> dot product == cosine similarity.
    doc_mat = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    qry_mat = qry_vecs / np.linalg.norm(qry_vecs, axis=1, keepdims=True)

    ids = [f"d{i}" for i in range(N_DOCS)]
    # Exact ground truth for every query (as id STRINGS, to match chroma results).
    truth = [[ids[i] for i in brute_force_topk(q, doc_mat, K)] for q in qry_mat]

    results = []   # (M, efC, ef, insert_s, query_ms, recall)

    # ---- Sweep build configs x query ef --------------------------------------
    for M, efC in BUILD_CONFIGS:
        for ef in QUERY_EFS:
            cname = f"hnsw_m{M}_efc{efC}_ef{ef}"
            try:
                client.delete_collection(cname)
            except Exception:
                pass
            col = client.create_collection(
                name=cname,
                metadata={"hnsw:space": "cosine",
                          "hnsw:M": M,
                          "hnsw:construction_ef": efC,
                          "hnsw:search_ef": ef},
            )
            # --- timed INSERT (embeddings precomputed; only graph build is measured)
            t0 = time.perf_counter()
            col.add(ids=ids, embeddings=doc_vecs.tolist())
            insert_s = time.perf_counter() - t0

            # --- timed QUERIES + recall vs numpy ground truth
            hits, total = 0, 0
            t0 = time.perf_counter()
            for q, gt in zip(qry_mat, truth):
                res = col.query(query_embeddings=[q.tolist()], n_results=K)
                got = set(res["ids"][0])
                hits += len(got & set(gt))
                total += K
            query_ms = (time.perf_counter() - t0) / N_QUERIES * 1000
            recall = hits / total

            results.append((M, efC, ef, insert_s, query_ms, recall))
            client.delete_collection(cname)
            console.print(f"  done M={M:<3} efC={efC:<4} ef={ef:<4} "
                          f"insert={insert_s*1000:7.1f}ms  query={query_ms:6.2f}ms  "
                          f"recall@{K}={recall:.3f}")

    # ---- Results table ---------------------------------------------------------
    t = Table(title=f"HNSW tuning - {N_DOCS} docs, dim={doc_vecs.shape[1]}, recall@{K} vs numpy brute force",
              title_style="bold cyan")
    t.add_column("M", justify="right")
    t.add_column("efConstruction", justify="right")
    t.add_column("ef (query)", justify="right")
    t.add_column("insert time", justify="right")
    t.add_column("avg query", justify="right")
    t.add_column(f"recall@{K}", justify="right")
    best_recall = max(r[5] for r in results)
    for M, efC, ef, ins, qms, rec in results:
        style = "bold green" if rec == best_recall else None
        t.add_row(str(M), str(efC), str(ef),
                  f"{ins*1000:.0f} ms", f"{qms:.2f} ms",
                  f"{rec:.3f}", style=style)
    console.print(t)

    # ---- Recommendation -----------------------------------------------------------
    # Heuristic: pick the cheapest config that reaches >= 98% of the best recall.
    threshold = 0.98 * best_recall
    affordable = [r for r in results if r[5] >= threshold]
    pick = min(affordable, key=lambda r: (r[4], r[3]))   # fastest query, then fastest insert
    M, efC, ef, ins, qms, rec = pick
    console.print(Panel(
        f"[bold]Recommendation:[/bold] M={M}, efConstruction={efC}, search_ef={ef}\n"
        f"  recall@{K}={rec:.3f} (best={best_recall:.3f}), avg query={qms:.2f}ms, "
        f"insert={ins*1000:.0f}ms\n\n"
        "[dim]Reading the table:\n"
        "- Higher M / efConstruction -> denser graph -> better recall, SLOWER inserts, bigger memory.\n"
        "- Higher query ef -> better recall, SLOWER queries (beam walks more candidates).\n"
        "- At ~500 docs everything is fast; the trade-offs only bite past ~100k vectors.\n"
        "- Rule of thumb: start M=16/efC=200 for text RAG; raise query ef when recall < 0.9.[/]",
        title="How to read this", border_style="blue"))

    client.close()   # release SQLite handles so the temp dir can be removed on Windows
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except Exception:
        pass
    console.rule("[bold green]HNSW tuning complete [OK][/]")


if __name__ == "__main__":
    main()
