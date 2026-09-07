import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import tempfile, os, traceback
import numpy as np
import chromadb

print("chromadb", chromadb.__version__)
d = tempfile.mkdtemp()
client = chromadb.PersistentClient(path=d)

# Test 1: HNSW params via metadata + list metadata + embeddings add/query
try:
    col = client.create_collection(
        name="probe",
        metadata={"hnsw:space": "cosine", "hnsw:M": 4, "hnsw:efConstruction": 16},
    )
    print("T1 create w/ hnsw M/efConstruction OK")
except Exception as e:
    print("T1 FAIL", repr(e)); traceback.print_exc()

try:
    vecs = np.random.rand(5, 384).astype("float32").tolist()
    col.add(ids=[f"i{i}" for i in range(5)],
            documents=[f"doc {i}" for i in range(5)],
            metadatas=[{"category": "a", "year": 2020, "tags": ["x", "y"],
                        "price": 1.5, "active": True} for _ in range(5)],
            embeddings=vecs)
    print("T2 add w/ list metadata OK, count=", col.count())
except Exception as e:
    print("T2 FAIL", repr(e)); traceback.print_exc()

try:
    q = np.random.rand(384).astype("float32").tolist()
    r = col.query(query_embeddings=[q], n_results=3)
    print("T3 query query_embeddings OK ids=", r["ids"][0])
except Exception as e:
    print("T3 FAIL", repr(e)); traceback.print_exc()

try:
    r = col.query(query_embeddings=[np.random.rand(384).astype('float32').tolist()],
                  n_results=3, where={"category": {"$in": ["a"]}},
                  parameters={"ef": 50})
    print("T4 query w/ $in + parameters ef OK ids=", r["ids"][0])
except Exception as e:
    print("T4 FAIL", repr(e)); traceback.print_exc()

try:
    g = col.get(where={"tags": {"$in": ["x"]}})
    print("T5 get $in on LIST field OK ids=", g["ids"])
except Exception as e:
    print("T5 FAIL (list $in)", repr(e))

try:
    g = col.get(where={"$and": [{"category": "a"}, {"year": {"$gte": 2020}}]})
    print("T6 $and OK ids=", g["ids"])
except Exception as e:
    print("T6 FAIL", repr(e))

# upsert / update
try:
    col.upsert(ids=["i0"], documents=["updated doc 0"],
               metadatas=[{"category": "b", "year": 2024, "tags": ["z"],
                           "price": 9.9, "active": False}],
               embeddings=[np.random.rand(384).astype('float32').tolist()])
    print("T7 upsert OK")
except Exception as e:
    print("T7 FAIL", repr(e))

# close + reopen
try:
    client.close()
    print("T8 close OK")
    c2 = chromadb.PersistentClient(path=d)
    c2col = c2.get_collection("probe")
    print("T9 reopen+get_collection OK count=", c2col.count())
    c2.close()
except Exception as e:
    print("T8/T9 FAIL", repr(e)); traceback.print_exc()
