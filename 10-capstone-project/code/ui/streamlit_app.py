"""Papeer — live Streamlit UI over the papeer package.

Layout:
- Sidebar: top-k slider, score threshold, LLM provider display, index rebuild button.
- Main: query box -> hybrid retrieval results (score + source metadata) -> generated answer.

Production behaviors (this is a capstone, not a toy):
- Never crashes on startup: if no LLM provider is reachable we show a friendly hint and
  keep the UI usable in retrieval-only mode.
- If the index is missing we offer a one-click build instead of a stack trace.
- Every LLM call is wrapped: network/auth failures surface as a hint, not an exception.

Run (from the repo root):
    uv run streamlit run 10-capstone-project/code/ui/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import pathlib

# Windows UTF-8 guard (emoji/box-drawing in consoles & logs).
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make `papeer` (sibling package) and `shared` (project root) importable.
# This file lives at: <root>/10-capstone-project/code/ui/streamlit_app.py
_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

import requests  # noqa: E402
import streamlit as st  # noqa: E402

from papeer.config import CHUNKS_JSON, get_llm, get_llm_provider_name  # noqa: E402
from papeer.ingestion import build_index  # noqa: E402
from papeer.retrieval import retrieve  # noqa: E402


# --------------------------------------------------------------------------- #
# Provider reachability (cheap HTTP ping — never a full LLM call at startup)
# --------------------------------------------------------------------------- #
def check_llm_provider() -> tuple[bool, str]:
    """Return (reachable, detail). Tries the active provider's /models endpoint."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    yolo_key = os.getenv("YOLO_AUTO_API_KEY", "").strip()
    try:
        if openai_key:
            url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/models"
            r = requests.get(url, headers={"Authorization": f"Bearer {openai_key}"}, timeout=5)
            return r.status_code < 500, f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4o-mini')})"
        if yolo_key:
            base = os.getenv("YOLO_AUTO_BASE_URL", "https://yolo-auto.com/v1").rstrip("/")
            r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {yolo_key}"}, timeout=5)
            return r.status_code < 500, f"Yolo-Auto ({os.getenv('YOLO_AUTO_MODEL', 'qwen3.8-27b')})"
        r = requests.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags", timeout=5)
        return r.ok, f"Ollama local ({os.getenv('OLLAMA_MODEL', 'llama3.1')})"
    except Exception as e:  # noqa: BLE001 - unreachable must never crash the UI
        return False, f"provider check failed: {type(e).__name__}"


# --------------------------------------------------------------------------- #
# Answer generation (always wrapped — a dead LLM degrades to retrieval-only)
# --------------------------------------------------------------------------- #
ANSWER_PROMPT = """You are Papeer, a precise research assistant. Answer the question using ONLY
the numbered context passages below. Cite passages inline as [n]. If the context does not
contain the answer, say so plainly — do not invent facts.

{context}

Question: {question}

Answer:"""


def generate_answer(question: str, hits: list[dict]) -> str | None:
    """Call the LLM with retrieved context. Returns the answer or None on failure."""
    if not hits:
        return None
    context = "\n\n".join(f"[{h['rank']}] (source: {h['source']})\n{h['text']}" for h in hits)
    llm = get_llm(temperature=0.0)
    return llm.invoke(ANSWER_PROMPT.format(context=context, question=question)).content


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Papeer — RAG Research Assistant", page_icon="📄", layout="wide")
    st.title("📄 Papeer — Capstone RAG Research Assistant")
    st.caption("Hybrid retrieval (dense + BM25 → RRF) → cross-encoder rerank → grounded, cited answers.")

    # --- Sidebar controls --------------------------------------------------- #
    with st.sidebar:
        st.header("⚙️ Controls")
        k = st.slider("Top-k results", min_value=1, max_value=10, value=3)
        threshold = st.slider("Minimum rerank score", min_value=-10.0, max_value=10.0,
                              value=-10.0, step=0.1,
                              help="Hide passages whose cross-encoder score is below this.")
        st.divider()
        st.subheader("🤖 LLM provider")
        if "provider" not in st.session_state:
            st.session_state.provider = check_llm_provider()
        reachable, detail = st.session_state.provider
        if st.button("🔁 Re-check provider", use_container_width=True):
            st.session_state.provider = check_llm_provider()
            st.rerun()
        if reachable:
            st.success(f"{detail}\n\n{get_llm_provider_name()}")
        else:
            st.warning(
                f"**LLM not reachable** ({detail}).\n\n"
                "The UI still works in **retrieval-only mode**. To enable answers, set "
                "`YOLO_AUTO_API_KEY` or `OPENAI_API_KEY` in `.env` (or run Ollama locally), "
                "then restart the app."
            )
        st.divider()
        if st.button("🔄 Rebuild index", use_container_width=True):
            with st.spinner("Ingesting corpus (embedding all documents)…"):
                try:
                    summary = build_index(verbose=False)
                    st.success(f"Indexed {summary['total_chunks']} chunks from {summary['files']} files.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Index build failed: {e}")

    # --- Index guard --------------------------------------------------------- #
    if not CHUNKS_JSON.exists():
        st.info("👋 **No index yet.** Click **Rebuild index** in the sidebar to ingest the "
                "sample corpus (takes ~1 minute on first run).")
        return

    # --- Query + retrieval ---------------------------------------------------- #
    question = st.text_input("❓ Ask a question about the corpus",
                             placeholder="e.g. What navigation technology does the Falcon AMR use?")
    if not question.strip():
        st.caption("Type a question above — try “Which vector index builds a multi-layer graph?”")
        return

    with st.spinner("Retrieving (dense + BM25 → RRF → cross-encoder rerank)…"):
        hits = retrieve(question, top_k=k)
    hits = [h for h in hits if h["score"] >= threshold]

    st.subheader(f"🔎 Top-{k} results {f'(filtered ≥ {threshold:g})' if threshold > -10 else ''}")
    if not hits:
        st.warning("No passages passed the score threshold — lower it in the sidebar.")
    for h in hits:
        with st.container(border=True):
            st.markdown(
                f"**#{h['rank']}** · score `{h['score']:.3f}` · "
                f"📁 `{h['source']}` · page {h.get('page')}"
            )
            with st.expander("Passage"):
                st.text(h["text"])

    # --- Generated answer ------------------------------------------------------ #
    st.subheader("🤖 Generated answer")
    if not reachable:
        st.info("Answer generation disabled — no LLM provider reachable (see sidebar). "
                "Retrieval results above are still fully functional.")
        return
    if not hits:
        st.caption("Nothing retrieved, so there is nothing to ground an answer in.")
        return
    with st.spinner("Generating grounded answer…"):
        try:
            answer = generate_answer(question, hits)
        except Exception as e:  # noqa: BLE001 - LLM failures degrade, never crash
            st.error(
                f"Answer generation failed ({type(e).__name__}: {e}).\n\n"
                "Check your API key / network in `.env`, then retry. "
                "Retrieval results above remain valid."
            )
            return
    if answer:
        st.markdown(answer)
    else:
        st.warning("The model returned an empty answer.")


if __name__ == "__main__":
    main()
