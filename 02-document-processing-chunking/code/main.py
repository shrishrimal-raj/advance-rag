"""Module 2 — Document Processing & Chunking Lab.

1. Ingests every file in data/samples/ with a format-appropriate loader (+ metadata)
2. Compares 4 splitting strategies on the same document
3. Demonstrates metadata filtering

Run from project root:
    uv run python 02-document-processing-chunking/code/main.py
"""
import sys
import json
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
SAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "samples"


# ---------------------------------------------------------------- Step 1: LOADERS
def load_all_samples():
    """Load every sample file with the right loader; attach source+format metadata."""
    from langchain_community.document_loaders import (
        TextLoader,
        CSVLoader,
        JSONLoader,
        UnstructuredMarkdownLoader,
    )

    loader_map = {
        ".txt": (TextLoader, {"encoding": "utf-8"}),
        ".csv": (CSVLoader, {}),
        ".json": (JSONLoader, {"jq_schema": ".", "text_content": False}),
        ".md": (UnstructuredMarkdownLoader, {}),
    }

    docs = []
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.suffix not in loader_map:
            continue
        loader_cls, kwargs = loader_map[path.suffix]
        try:
            loaded = loader_cls(str(path), **kwargs).load()
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to load {path.name}: {e}[/yellow]")
            continue
        for d in loaded:
            # JSONLoader may hand back dicts — serialize so downstream sees plain text
            if isinstance(d.page_content, (dict, list)):
                d.page_content = json.dumps(d.page_content, indent=2)
            d.metadata.setdefault("source", str(path))
            d.metadata["format"] = path.suffix.lstrip(".")
        docs.extend(loaded)
        console.print(f"[green]✓ {path.name:<28} -> {len(loaded)} document(s)[/green]")
    return docs


# ---------------------------------------------------------------- Step 2: SPLITTERS
def custom_section_split(docs, max_chars=600):
    """Custom splitter: split on blank lines (sections); merge tiny sections."""
    from langchain_core.documents import Document

    out = []
    for doc in docs:
        sections = [s.strip() for s in doc.page_content.split("\n\n") if s.strip()]
        buf, meta = "", dict(doc.metadata)
        for sec in sections:
            if len(buf) + len(sec) + 2 > max_chars and buf:
                out.append(Document(page_content=buf, metadata=dict(meta)))
                buf = ""
            buf = f"{buf}\n\n{sec}" if buf else sec
        if buf:
            out.append(Document(page_content=buf, metadata=dict(meta)))
    return out


def md_header_split(docs):
    """Markdown header split (version-safe): split_text + re-derive header metadata."""
    from langchain_core.documents import Document
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
    out = []
    for d in docs:
        for sec in splitter.split_text(d.page_content):
            # Version-safe: newer langchain returns Document objects, older ones return str
            if hasattr(sec, "page_content"):
                text, extra_meta = sec.page_content, dict(sec.metadata)
            else:
                text, extra_meta = sec, {}
            meta = {**d.metadata, **extra_meta}
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("## "):
                    meta["h2"] = line[3:]
                elif line.startswith("# "):
                    meta["h1"] = line[2:]
            out.append(Document(page_content=text, metadata=meta))
    return out


def compare_splitters():
    """Run 4 strategies on the same markdown doc and print stats."""
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    md_path = SAMPLES_DIR / "vector_db_notes.md"
    # Use RAW text so every strategy (incl. header-based) sees identical input.
    # (UnstructuredMarkdownLoader re-parses markdown and strips '#' markers.)
    from langchain_core.documents import Document as _Doc
    docs = [_Doc(page_content=md_path.read_text(encoding="utf-8"),
                 metadata={"source": str(md_path), "format": "md"})]

    strategies = {
        "CharacterTextSplitter (hard cuts)": lambda ds: CharacterTextSplitter(
            chunk_size=300, chunk_overlap=0
        ).split_documents(ds),
        "RecursiveCharacterTextSplitter (default)": lambda ds: RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        ).split_documents(ds),
        "MarkdownHeaderTextSplitter (structure)": md_header_split,
        "Custom section splitter (blank lines)": lambda ds: custom_section_split(ds),
    }

    table = Table(title="Splitter Strategy Comparison — vector_db_notes.md")
    table.add_column("Strategy", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Min chars", justify="right")
    table.add_column("Max chars", justify="right")
    table.add_column("Avg chars", justify="right")
    table.add_column("First chunk preview")

    for name, fn in strategies.items():
        chunks = fn(docs)
        sizes = [len(c.page_content) for c in chunks]
        table.add_row(
            name,
            str(len(chunks)),
            str(min(sizes)) if sizes else "-",
            str(max(sizes)) if sizes else "-",
            str(round(sum(sizes) / len(sizes))) if sizes else "-",
            (chunks[0].page_content[:70] + "...") if chunks else "-",
        )
    console.print(table)

    # Show what structure-preserving split captured in metadata
    md_chunks = strategies["MarkdownHeaderTextSplitter (structure)"](docs)
    console.print("\n[bold]Header metadata captured by MarkdownHeaderTextSplitter:[/bold]")
    for c in md_chunks[:4]:
        hdrs = {k: v for k, v in c.metadata.items() if k.startswith(("h1", "h2"))}
        console.print(f"  • {hdrs}  -> {len(c.page_content)} chars")
    return md_chunks


# ---------------------------------------------------------------- Step 3: FILTERING
def demo_metadata_filtering(all_docs):
    """Show that metadata attached at load time enables filtered access later."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(all_docs)
    console.rule("[bold cyan]Metadata filtering demo[/bold cyan]")
    for fmt in ["txt", "md", "csv", "json"]:
        subset = [c for c in chunks if c.metadata.get("format") == fmt]
        console.print(f"  format={fmt:<5} -> {len(subset):>3} chunks")
    console.print(
        "\n[dim]In Modules 5-6 these become retriever filters, e.g.\n"
        '  retriever.invoke(q, filters={"format": "csv"})  — search ONLY product data.[/dim]'
    )


def main():
    console.print(Panel("[bold]Module 2 — Document Processing & Chunking Lab[/bold]", border_style="blue"))

    console.rule("[bold]1) Ingestion with format-aware loaders[/bold]")
    all_docs = load_all_samples()
    console.print(f"[bold green]Total documents loaded: {len(all_docs)}[/bold green]\n")

    console.rule("[bold]2) Splitter strategy comparison[/bold]")
    compare_splitters()

    console.rule("[bold]3) Metadata filtering[/bold]")
    demo_metadata_filtering(all_docs)

    console.rule("[bold red]Recommendation cheat-sheet[/bold red]")
    console.print(
        """
• Plain prose / logs ............ RecursiveCharacterTextSplitter (~500 tokens, 10-15% overlap)
• Markdown / structured docs .... MarkdownHeaderTextSplitter -> recursive fallback for long sections
• Code .......................... Language-aware (AST) splitter, never plain character cuts
• Tables / CSV .................. Keep rows intact; one row per document
• Research papers / narrative ... Consider semantic chunking (Module 6+)
"""
    )


if __name__ == "__main__":
    main()
