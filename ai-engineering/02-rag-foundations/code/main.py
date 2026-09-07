import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The Knowledge Engine (Week 2 weekly build).

A context-aware RAG pipeline over a local Chroma vector store:
load -> chunk -> embed (local MiniLM) -> store (per-tenant collection)
-> retrieve top-k -> grounded answer (cloud LLM) with token accounting.

Run:
    python main.py --selftest
    python main.py --question "..." --tenant acme
"""
import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm, get_embeddings, get_llm_provider_name, CHROMA_DIR  # noqa: E402

CORPUS = {
    "acme": [
        "Acme refund policy: orders may be returned within 30 days for a full refund. Refunds are issued to the original payment method within 5 business days.",
        "Acme shipping: standard delivery takes 3 to 5 business days. Express delivery is available for an extra fee and arrives in 1 to 2 business days.",
        "Acme warranty: all hardware carries a 12 month limited warranty covering manufacturing defects.",
    ],
    "globex": [
        "Globex support hours are Monday to Friday, 9am to 6pm local time. Weekend support is available to enterprise customers only.",
        "Globex data retention: account data is retained for 90 days after cancellation, then permanently deleted.",
    ],
}

PROMPT = (
    "Answer the question using ONLY the context below. If the answer is not in "
    "the context, say you do not know. Cite the source.\n\nContext:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)


def build_index(tenant, texts):
    import chromadb
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts_, metas = [], []
    for i, t in enumerate(texts):
        for c in splitter.split_text(t):
            texts_.append(c)
            metas.append({"tenant": tenant, "source": f"{tenant}-doc{i}"})
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    cname = f"knowledge_{tenant}"
    try:
        client.delete_collection(cname)
    except Exception:
        pass
    col = Chroma(client=client, collection_name=cname, embedding_function=get_embeddings())
    col.add_texts(texts_, metadatas=metas)
    return col, len(texts_)


def answer(question, tenant="acme", k=3):
    col, _ = build_index(tenant, CORPUS.get(tenant, []))
    hits = col.similarity_search_with_relevance_scores(question, k=k)
    context = "\n".join(f"- [{h.metadata.get('source')}] {h.page_content}" for h, _ in hits)
    llm = get_llm()
    res = llm.invoke(PROMPT.format(context=context, question=question))
    return res.content, hits, getattr(res, "usage_metadata", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="What is Acme's refund policy?")
    ap.add_argument("--tenant", default="acme")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    print(f"[Knowledge Engine] provider={get_llm_provider_name()}")
    try:
        ans, hits, usage = answer(args.question, args.tenant, args.k)
    except Exception as e:
        print(f"[Knowledge Engine] LLM/embedding unavailable ({type(e).__name__}: {e})")
        print("Hint: set YOLO_AUTO_API_KEY in .env (cloud LLM). Exiting gracefully.")
        return 0
    print("\n=== Retrieved context ===")
    for h, score in hits:
        print(f"  ({score:.3f}) [{h.metadata.get('source')}] {h.page_content[:90]}")
    print("\n=== Answer ===\n" + ans.strip())
    if usage:
        print(f"\n[tokens] total={usage.get('total_tokens')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
