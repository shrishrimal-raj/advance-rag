import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 2 - raw vector DB API (no LangChain).

Uses the Chroma client + sentence-transformers directly. Same retrieval as
main.py but with zero framework abstraction - see exactly what the vector DB
does. Runs offline (cached MiniLM); no LLM needed.
"""
import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import CHROMA_DIR  # noqa: E402

CORPUS = {
    "acme": [
        "Acme refund policy: orders may be returned within 30 days for a full refund.",
        "Acme shipping: standard delivery takes 3 to 5 business days.",
        "Acme warranty: all hardware carries a 12 month limited warranty.",
    ],
    "globex": [
        "Globex support hours are Monday to Friday, 9am to 6pm local time.",
        "Globex data retention: account data is retained for 90 days after cancellation.",
    ],
}


def embed(model, texts):
    return model.encode(texts, normalize_embeddings=True).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="what is the refund policy")
    ap.add_argument("--tenant", default="acme")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    import chromadb

    texts = CORPUS.get(args.tenant, [])
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    cname = f"raw_{args.tenant}"
    try:
        client.delete_collection(cname)
    except Exception:
        pass
    col = client.get_or_create_collection(cname)
    ids = [str(i) for i in range(len(texts))]
    metas = [{"tenant": args.tenant, "source": f"{args.tenant}-{i}"} for i in range(len(texts))]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embed(model, texts))

    res = col.query(
        query_embeddings=embed(model, [args.question])[0],
        n_results=args.k,
        where={"tenant": args.tenant},
    )
    print(f"=== Approach 2: raw Chroma retrieval (tenant={args.tenant}) ===")
    print(f"question = {args.question!r}")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"  dist={dist:.4f} [{meta['source']}] {doc[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
