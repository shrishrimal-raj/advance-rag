"""
Module 3 - Approach 2: Cloud embeddings vs local embeddings, head-to-head.

Run from the project root:
    uv run python 03-embeddings-vector-representations/code/approach_2_cloud_embeddings.py

What this script does:
  1. Detects which embedding provider is available:
       - OPENAI_API_KEY set        -> OpenAI text-embedding-3-small (cloud)
       - no key                    -> explains why Yolo-Auto is NOT an option for
                                      embeddings (it exposes chat completions only,
                                      no /embeddings endpoint), then falls back to
                                      the local all-MiniLM-L6-v2 model.
  2. Embeds a small batch of RAG-domain texts and measures:
       - output dimensionality
       - wall-clock latency (first call incl. warm-up, then steady-state)
  3. Prints a cost/latency/privacy comparison table (cloud vs local) so you can
     make an informed choice for your own project.

Never crashes without keys: if the cloud call fails at runtime (bad key, no
internet), it prints the error and degrades to the local model instead.
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import pathlib
import time

# Make the project root importable so we can reuse shared.config from any folder.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from shared.config import get_embeddings

console = Console()

SAMPLE_TEXTS = [
    "Retrieval augmented generation grounds answers in external documents.",
    "A vector database stores embeddings for fast semantic search.",
    "Chunking splits long documents into smaller pieces before embedding.",
    "Cosine similarity measures the angle between two vectors.",
    "Hybrid retrieval combines dense vectors with BM25 keyword scores.",
]


def build_cloud_embeddings():
    """OpenAI embeddings client (only provider with a real /embeddings endpoint)."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        max_retries=1,
    )


def build_local_embeddings():
    """Local sentence-transformers MiniLM - free, private, no key needed.

    Reuses the shared factory so .env (LOCAL_EMBEDDING_MODEL) stays the single
    source of truth. Note: if EMBEDDING_PROVIDER=openai is set in .env this would
    return the cloud client - in that case the 'local baseline' label below is
    whatever provider .env selects, which is exactly what your pipeline uses.
    """
    return get_embeddings()


def timed_embed(emb, name: str):
    """Embed SAMPLE_TEXTS twice; report first-call (warm-up) and steady-state ms."""
    t0 = time.perf_counter()
    vecs = emb.embed_documents(SAMPLE_TEXTS)
    first_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    emb.embed_documents(SAMPLE_TEXTS)
    steady_ms = (time.perf_counter() - t0) * 1000

    arr = np.asarray(vecs, dtype=np.float32)
    console.print(f"  [bold]{name}[/bold]")
    console.print(f"    dimensionality : {arr.shape[1]}")
    console.print(f"    first call     : {first_ms:,.0f} ms  (includes model load / TLS handshake)")
    console.print(f"    steady state   : {steady_ms:,.0f} ms  ({len(SAMPLE_TEXTS)} texts)")
    return arr.shape[1], first_ms, steady_ms


def print_cost_table(cloud_ok: bool) -> None:
    t = Table(title="Cloud vs local embeddings - decision table", title_style="bold magenta")
    t.add_column("Factor", style="bold cyan")
    t.add_column("OpenAI text-embedding-3-small (cloud)")
    t.add_column("all-MiniLM-L6-v2 (local)")
    t.add_row("Dimensions", "1536 (can request fewer via `dimensions=`)", "384")
    t.add_row("Cost per 1M tokens", "~$0.07 (input)", "$0 (your electricity bill)")
    t.add_row("Latency (typical)", "100-500 ms network round-trip", "5-50 ms after warm-up")
    t.add_row("Privacy", "Text leaves your machine", "Fully on-prem")
    t.add_row("Rate limits", "Tier-based RPM/TPM caps", "None (bounded by your CPU/GPU)")
    t.add_row("Quality ceiling", "Higher on hard semantic tasks", "Good for general English")
    t.add_row("Offline use", "No", "Yes")
    console.print(t)
    note = ("[green]Cloud path is active in this run.[/]" if cloud_ok else
            "[yellow]Cloud path NOT active (no OPENAI_API_KEY) - results above are local-only.[/]")
    console.print(note)
    console.print(Panel(
        "[bold]Rule of thumb:[/]\n"
        "  * Prototype / private data / tight budget  -> local MiniLM or BGE.\n"
        "  * Production quality bar + data may leave the box -> OpenAI/Cohere.\n"
        "  * Either way: pick ONE model, normalize vectors, and rebuild the index\n"
        "    whenever you swap models (vectors are not portable across models).",
        title="Cost & choice guidance", border_style="cyan"))


def main() -> None:
    console.rule("[bold cyan]Module 3 - Approach 2: Cloud vs Local Embeddings[/]")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    yolo_key = os.getenv("YOLO_AUTO_API_KEY", "").strip()

    # ---- Provider selection ------------------------------------------------
    if openai_key:
        console.print("[bold]Step 1:[/bold] Provider detection")
        console.print("  * OPENAI_API_KEY found -> using OpenAI cloud embeddings.")
        if yolo_key:
            console.print("  * YOLO_AUTO_API_KEY also present, but ignored here: "
                          "Yolo-Auto is an OpenAI-compatible [italic]chat[/italic] gateway "
                          "(qwen3.8-27b) and exposes NO /embeddings endpoint, so it cannot "
                          "produce vectors. Cloud embeddings require a real embeddings API "
                          "(OpenAI, Cohere, Voyage, ...).")
    else:
        console.print("[bold]Step 1:[/bold] Provider detection")
        console.print(Panel(
            "No [bold]OPENAI_API_KEY[/bold] found.\n\n"
            "Why not Yolo-Auto? It is an OpenAI-compatible [italic]chat completions[/italic] "
            "provider (model qwen3.8-27b) - it has [red]no /embeddings endpoint[/red], so it "
            "cannot generate vectors. The cloud path therefore requires a provider with a real "
            "embeddings API (OpenAI shown here).\n\n"
            "[yellow]Falling back to the local all-MiniLM-L6-v2 model so this lab still runs.[/]",
            title="Graceful fallback", border_style="yellow"))

    # ---- Cloud attempt (with runtime degradation) ---------------------------
    cloud_ok = False
    if openai_key:
        try:
            cloud_emb = build_cloud_embeddings()
            timed_embed(cloud_emb, "OpenAI text-embedding-3-small")
            cloud_ok = True
        except Exception as exc:
            console.print(Panel(
                f"[red]Cloud embedding call failed:[/]\n{exc}\n\n"
                "[yellow]Check the key / network, then re-run. Falling back to local model.[/]",
                title="Cloud unavailable", border_style="red"))

    # ---- Local (always runs: baseline, or sole path when no key) ------------
    console.print("\n[bold]Step 2:[/bold] Local baseline (sentence-transformers MiniLM)")
    try:
        local_emb = build_local_embeddings()
        timed_embed(local_emb, "all-MiniLM-L6-v2 (local)")
    except Exception as exc:
        console.print(Panel(
            f"[red]Could not load the local embedding model:[/]\n{exc}\n\n"
            "[yellow]First run downloads all-MiniLM-L6-v2 (~90 MB) from Hugging Face.\n"
            "Check your internet connection, then re-run.[/]",
            title="Setup issue", border_style="red"))
        return

    # ---- Comparison ----------------------------------------------------------
    console.print("\n[bold]Step 3:[/bold] Cost / latency / privacy comparison")
    print_cost_table(cloud_ok)

    console.rule("[bold green]Done - compare the numbers above against your own budget & privacy needs.[/]")


if __name__ == "__main__":
    main()
