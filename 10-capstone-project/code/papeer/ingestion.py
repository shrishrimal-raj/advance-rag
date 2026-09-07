"""Phase 1 — Multi-format ingestion.

Loads every supported document in the corpus, normalizes to text blocks,
chunks with metadata, embeds, persists to a persistent Chroma collection, and writes a
JSON sidecar (chunks.json) that the BM25 retriever reads.

Design notes (industry practice):
- Format-aware loaders -> normalized text blocks, so downstream code is format-agnostic.
  We use *native* loaders (pypdf / csv / json stdlib) instead of langchain_community:
  that package's import chain eagerly pulls in sentence_transformers + torch (~1 min on
  a laptop), which would tax every ingestion run and every test for zero quality gain.
- Stable chunk IDs (hash of source+index) => re-ingestion is idempotent (no dup vectors).
- Metadata (source/page/chunk_index/doc_type) powers citations, filtering, and later RBAC.
- We rebuild the collection from scratch each run for a clean, reproducible index.

Run:  uv run python 10-capstone-project/code/papeer/ingestion.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import pathlib

_CODE_DIR = str(pathlib.Path(__file__).resolve().parents[1])   # .../code
_ROOT = str(pathlib.Path(__file__).resolve().parents[3])       # project root
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from rich.console import Console
from rich.table import Table

from papeer.config import (
    CHUNKS_JSON,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    get_embeddings,
    supported_files,
)

# Force UTF-8 output so emoji/box-drawing render correctly on Windows consoles (cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

console = Console()


def _load_file(path: pathlib.Path) -> list[str]:
    """Return the raw text blocks for a single file, using a format-aware loader.

    Native loaders (pypdf / csv / json stdlib) keep the import graph light. Each returns
    one block per logical unit (PDF page, CSV row, whole JSON doc, whole text file) so the
    `page` metadata downstream means something real. A quirky file never crashes the job.
    """
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return [(page.extract_text() or "") for page in reader.pages]
        if ext == ".csv":
            with path.open(encoding="utf-8", newline="") as fh:
                rows = [",".join(row) for row in csv.reader(fh) if any(c.strip() for c in row)]
            return rows
        if ext == ".json":
            # Flatten nested JSON into readable lines so it embeds meaningfully.
            data = json.loads(path.read_text(encoding="utf-8"))
            return [_flatten_json(data)]
        # .txt / .md -> plain text
        return [path.read_text(encoding="utf-8")]
    except Exception as e:  # noqa: BLE001 - graceful degradation per-file
        console.print(f"[yellow]⚠️  loader failed for {path.name}: {e}; using raw text[/yellow]")
        return [path.read_text(encoding="utf-8", errors="ignore")]


def _flatten_json(obj, prefix: str = "") -> str:
    """Flatten nested JSON into 'key: value' lines (good for embedding + reading)."""
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                lines.append(_flatten_json(v, key))
            else:
                lines.append(f"{key}: {v}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.append(_flatten_json(v, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {obj}")
    return "\n".join(l for l in lines if l.strip())


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Lightweight recursive splitter: paragraph -> sentence -> word, with overlap.

    Deliberately NOT langchain_text_splitters: its package __init__ imports
    sentence_transformers (-> torch), which costs ~1 minute on this machine for no
    quality difference on a small corpus. Behavior mirrors RecursiveCharacterTextSplitter:
    prefer natural boundaries, hard-split only when a unit still exceeds chunk_size, and
    carry `overlap` characters from the previous chunk into the next.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    def _units(s: str) -> list[str]:
        paras = [p for p in re.split(r"\n\s*\n", s) if p.strip()]
        if all(len(p) <= chunk_size for p in paras):
            return paras
        units: list[str] = []
        for p in paras:
            if len(p) <= chunk_size:
                units.append(p)
                continue
            units.extend(x for x in re.split(r"(?<=[.!?])\s+", p) if x.strip())
        out: list[str] = []
        for u in units:
            if len(u) <= chunk_size:
                out.append(u)
                continue
            cur = ""
            for w in u.split(" "):
                if cur and len(cur) + 1 + len(w) > chunk_size:
                    out.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                out.append(cur)
        return out

    chunks: list[str] = []
    cur = ""
    for u in _units(text):
        candidate = f"{cur}\n{u}" if cur else u
        if len(candidate) <= chunk_size:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
        tail = cur[-overlap:] if overlap and cur else ""
        cur = f"{tail}{u}".strip()
    if cur.strip():
        chunks.append(cur)
    return [c for c in (c.strip() for c in chunks) if c]


def _stable_id(source: str, idx: int) -> str:
    """Deterministic chunk id => idempotent upserts on re-ingestion."""
    return hashlib.md5(f"{source}::{idx}".encode("utf-8")).hexdigest()[:16]


def build_index(verbose: bool = True) -> dict:
    """Ingest the whole corpus. Returns a summary dict (counts, total chunks)."""
    from chromadb import PersistentClient

    files = supported_files()
    if not files:
        raise FileNotFoundError(f"No supported files found in {DATA_DIR_HINT()}")

    all_docs: list[dict] = []   # {id, text, source, page, chunk_index, doc_type}
    per_file: dict[str, int] = {}

    for path in files:
        blocks = _load_file(path)
        doc_type = path.suffix.lstrip(".").upper()
        n_chunks = 0
        for page_no, block in enumerate(blocks, start=1):
            if not block.strip():
                continue
            for idx, chunk in enumerate(_split_text(block, CHUNK_SIZE, CHUNK_OVERLAP)):
                all_docs.append({
                    "id": _stable_id(path.name, f"{page_no}-{idx}"),
                    "text": chunk,
                    "source": path.name,
                    "page": page_no,
                    "chunk_index": idx,
                    "doc_type": doc_type,
                })
                n_chunks += 1
        per_file[path.name] = n_chunks

    # --- Embed everything in one batch (fast) ------------------------------- #
    embeddings = get_embeddings()
    texts = [d["text"] for d in all_docs]
    vecs = embeddings.embed_documents(texts)

    # --- Persist to Chroma (rebuild for a clean, reproducible index) -------- #
    client = PersistentClient(str(CHROMA_DIR))
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    col = client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[d["id"] for d in all_docs],
        documents=texts,
        metadatas=[{k: d[k] for k in ("source", "page", "chunk_index", "doc_type")} for d in all_docs],
        embeddings=vecs,
    )

    # --- Write the BM25 sidecar -------------------------------------------- #
    CHUNKS_JSON.write_text(json.dumps(all_docs, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {"files": len(files), "total_chunks": len(all_docs), "per_file": per_file}

    if verbose:
        table = Table(title="📥 Papeer Ingestion Summary")
        table.add_column("File", style="cyan")
        table.add_column("Chunks", justify="right")
        for name, n in per_file.items():
            table.add_row(name, str(n))
        console.print(table)
        console.print(f"[green]✔ Indexed {len(all_docs)} chunks from {len(files)} files "
                      f"→ {CHROMA_COLLECTION}[/green]")
    return summary


def DATA_DIR_HINT() -> str:
    from .config import DATA_DIR, USER_DOCS_DIR
    return f"{DATA_DIR} (or {USER_DOCS_DIR})"


if __name__ == "__main__":
    build_index()
