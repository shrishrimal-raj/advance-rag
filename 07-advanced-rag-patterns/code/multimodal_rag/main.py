"""
Multi-Modal RAG (Cross-Modal Retrieval)
========================================
Pattern: Create images -> embed with CLIP -> store alongside text in Chroma
         -> cross-modal queries (text->image, image->text).

Problem solved: Real-world data includes images. Text-only RAG cannot retrieve
or reason about visual content. Multi-Modal RAG uses a shared embedding space
(CLIP) where both text and images live, enabling cross-modal search.

IMPORTANT: This script attempts to download clip-ViT-B-32 (~600 MB) on first run.
If the download fails, the model is unavailable, or any other error occurs, it
falls back to a text-only demo that explains the architecture.

PRODUCTION NOTE: clip-ViT-B-32 is a small model for demonstration. Production
systems use larger models like CLIP ViT-L/14 (~1.7 GB), SigLIP, or BLIP-2 for
better cross-modal alignment and fine-grained understanding.

Run from project root:
    uv run python 07-advanced-rag-patterns/code/multimodal_rag/main.py
"""

import sys
import pathlib

# Add project root to path so we can import shared.config
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))

# Windows console is cp1252; force UTF-8 output to avoid encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Stage 0: Create demo images with PIL
# ---------------------------------------------------------------------------
def create_demo_images() -> list[dict]:
    """
    Create 3 simple labeled images at runtime using PIL ImageDraw.
    Each image is a colored box with text label.
    
    Returns list of dicts: {"label": str, "image": PIL.Image, "description": str}
    """
    console.print(Panel("[bold cyan]Stage 0: Creating demo images with PIL[/bold cyan]"))

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        console.print("[red]Pillow not installed. Run: uv add pillow[/red]")
        return []

    # Define 3 images with different colors and labels
    image_specs = [
        {
            "label": "Vector DB",
            "color": (70, 130, 180),  # Steel blue
            "description": "A vector database stores high-dimensional embeddings for similarity search"
        },
        {
            "label": "BM25 Index",
            "color": (139, 69, 19),   # Saddle brown
            "description": "BM25 is a lexical ranking function used in traditional information retrieval"
        },
        {
            "label": "HNSW Graph",
            "color": (50, 178, 100),  # Medium sea green
            "description": "HNSW is a hierarchical navigable small world graph for approximate nearest neighbor search"
        },
    ]

    images = []
    for spec in image_specs:
        # Create a 300x150 image with the specified background color
        img = Image.new("RGB", (300, 150), color=spec["color"])
        draw = ImageDraw.Draw(img)

        # Draw a border
        draw.rectangle([5, 5, 294, 144], outline="white", width=3)

        # Try to use a larger font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 28)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # Center the text
        text = spec["label"]
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (300 - text_width) // 2
        y = (150 - text_height) // 2
        draw.text((x, y), text, fill="white", font=font)

        # Add a smaller subtitle
        try:
            small_font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            small_font = font

        subtitle = spec["description"][:40] + "..."
        draw.text((10, 120), subtitle, fill="white", font=small_font)

        images.append({
            "label": spec["label"],
            "image": img,
            "description": spec["description"]
        })
        console.print(f"  Created image: '{spec['label']}' ({spec['color']})")

    console.print(f"  Total: {len(images)} images created.\n")
    return images


# ---------------------------------------------------------------------------
# Stage 1: Load CLIP model for cross-modal embeddings
# ---------------------------------------------------------------------------
def load_clip_model():
    """
    Load the CLIP model for cross-modal embeddings.
    
    Uses sentence-transformers' wrapper around clip-ViT-B-32.
    This maps BOTH images and text into the same 512-dim vector space,
    enabling cross-modal similarity search.
    
    PRODUCTION NOTE: For production, use larger models:
    - sentence-transformers/clip-ViT-L-14 (~1.7 GB, better accuracy)
    - google/siglip-so400m-patch14-384 (state-of-the-art)
    - salesforce/blip-2 (for image captioning + retrieval)
    """
    console.print(Panel("[bold cyan]Stage 1: Loading CLIP model (clip-ViT-B-32)[/bold cyan]"))
    console.print("  [dim]This may download ~600 MB on first run...[/dim]\n")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/clip-ViT-B-32')
        console.print("  [green]CLIP model loaded successfully.[/green]\n")
        return model
    except Exception as e:
        console.print(f"[red]Failed to load CLIP model: {e}[/red]")
        console.print("[yellow]This is expected if the model isn't downloaded yet or[/yellow]")
        console.print("[yellow]if you're offline. Falling back to text-only demo.[/yellow]\n")
        return None


# ---------------------------------------------------------------------------
# Stage 2: Build unified vector store (images + text)
# ---------------------------------------------------------------------------
def build_multimodal_store(clip_model, images: list[dict], text_chunks: list[str]):
    """
    Store image embeddings and text chunk embeddings in the same Chroma collection.
    Because CLIP maps both modalities to the same space, cross-modal search works.
    """
    console.print(Panel("[bold cyan]Stage 2: Building unified vector store[/bold cyan]"))

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    import numpy as np

    # Embed images with CLIP
    console.print("  Embedding images...")
    image_embeddings = []
    for img_data in images:
        # CLIP expects PIL images or arrays
        emb = clip_model.encode(img_data["image"])
        image_embeddings.append(emb)

    # Embed text descriptions with CLIP (same model, same space!)
    console.print("  Embedding text chunks...")
    text_embeddings = [clip_model.encode(t) for t in text_chunks]

    # Combine all embeddings
    all_embeddings = image_embeddings + text_embeddings
    all_texts = [f"IMAGE: {img['label']} - {img['description']}" for img in images] + text_chunks
    all_metadatas = ([{"type": "image", "label": img["label"]} for img in images] +
                     [{"type": "text"} for _ in text_chunks])

    # Build Chroma store with pre-computed embeddings
    vs = Chroma(embedding_function=None)  # We'll provide embeddings directly
    vs.add_embeddings(
        embeddings=all_embeddings,
        documents=all_texts,
        metadatas=all_metadatas,
        ids=[f"doc_{i}" for i in range(len(all_embeddings))]
    )

    console.print(f"  Store built: {len(image_embeddings)} images + {len(text_chunks)} text chunks.")
    console.print(f"  All in the same {len(all_embeddings[0])}-dim CLIP space.\n")
    return vs


# ---------------------------------------------------------------------------
# Stage 3: Cross-modal queries
# ---------------------------------------------------------------------------
def cross_modal_query_text_to_image(clip_model, vector_store, text_query: str):
    """Query with TEXT, retrieve matching IMAGE."""
    console.print(Panel("[bold cyan]Stage 3a: Cross-modal query: TEXT -> IMAGE[/bold cyan]"))
    console.print(f"  Query: \"{text_query}\"\n")

    # Embed the text query with CLIP
    query_embedding = clip_model.encode(text_query)

    # Search in the unified store
    results = vector_store.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    console.print("  Results (text query retrieving images):")
    table = Table(show_lines=True)
    table.add_column("Rank", width=5)
    table.add_column("Result", width=50)
    table.add_column("Type", width=8)
    table.add_column("Distance", width=10)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ), 1):
        type_str = meta.get("type", "?")
        style = "magenta" if type_str == "image" else "cyan"
        table.add_row(str(i), doc[:47], f"[{style}]{type_str}[/{style}]", f"{dist:.4f}")

    console.print(table)
    return results


def cross_modal_query_image_to_text(clip_model, vector_store, image):
    """Query with IMAGE, retrieve matching TEXT."""
    console.print(Panel("[bold cyan]Stage 3b: Cross-modal query: IMAGE -> TEXT[/bold cyan]"))
    console.print(f"  Query: [Image of '{image.label}']\n")

    # Embed the image with CLIP
    query_embedding = clip_model.encode(image)

    # Search in the unified store
    results = vector_store.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    console.print("  Results (image query retrieving text):")
    table = Table(show_lines=True)
    table.add_column("Rank", width=5)
    table.add_column("Result", width=50)
    table.add_column("Type", width=8)
    table.add_column("Distance", width=10)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ), 1):
        type_str = meta.get("type", "?")
        style = "magenta" if type_str == "image" else "cyan"
        table.add_row(str(i), doc[:47], f"[{style}]{type_str}[/{style}]", f"{dist:.4f}")

    console.print(table)
    return results


# ---------------------------------------------------------------------------
# Fallback: Text-only demo explaining the architecture
# ---------------------------------------------------------------------------
def text_only_fallback(images: list[dict]):
    """
    When CLIP is unavailable, explain what WOULD happen in a multi-modal RAG system.
    This ensures the script always completes successfully.
    """
    console.print(Panel("[bold yellow]FALLBACK: Text-only Multi-Modal RAG Demo[/bold yellow]",
                        subtitle="CLIP model unavailable - showing architecture explanation"))

    console.print("""
[bold]What would happen with a working CLIP model:[/bold]

  1. [magenta]IMAGE EMBEDDING[/magenta]: Each image (like our 'Vector DB', 'BM25 Index',
     'HNSW Graph' diagrams) would be encoded into a 512-dimensional vector by CLIP's
     vision encoder (ViT-B/32).

  2. [cyan]TEXT EMBEDDING[/cyan]: Each text chunk would be encoded into the SAME
     512-dimensional space by CLIP's text encoder (transformer).

  3. [green]SHARED SPACE[/green]: Because both encoders are trained with contrastive
     learning on (image, caption) pairs, semantically similar content ends up close
     together REGARDLESS of modality. A photo of a dog is close to the text "a dog".

  4. [yellow]CROSS-MODAL RETRIEVAL[/yellow]:
     - Text query "diagram about vector databases" -> embed text -> 
       find nearest IMAGE embedding -> return the 'Vector DB' image
     - Image query [the HNSW Graph image] -> embed image ->
       find nearest TEXT embedding -> return the HNSW description chunk

  5. [bold]GENERATION[/bold]: Retrieved multimodal context (image + text) is fed to
     an LLM that can describe the image and answer questions about it.
""")

    # Show what our demo images would match
    console.print("[bold]Expected cross-modal matches for our demo images:[/bold]\n")
    table = Table(show_lines=True)
    table.add_column("Image", width=15)
    table.add_column("Would match text query:", width=40)
    table.add_column("Would match image query from:", width=30)

    matches = [
        ("Vector DB", "\"what stores embeddings for search?\"", "any DB diagram"),
        ("BM25 Index", "\"how does lexical ranking work?\"", "any search index image"),
        ("HNSW Graph", "\"approximate nearest neighbor algorithm?\"", "any graph structure image"),
    ]
    for img_label, text_q, img_q in matches:
        table.add_row(img_label, text_q, img_q)

    console.print(table)

    console.print("""
[bold]Production considerations:[/bold]
  - Use CLIP ViT-L/14 or SigLIP for better alignment
  - Pre-compute and cache image embeddings (expensive at inference time)
  - For fine-grained tasks (OCR, chart reading), combine CLIP with OCR/captioning
  - GPU strongly recommended (CPU inference is ~10x slower)
  - Consider BLIP-2 or LLaVA for image+text generation (not just retrieval)
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.print(Panel("[bold magenta]MULTI-MODAL RAG (CROSS-MODAL RETRIEVAL)[/bold magenta]",
                        subtitle="Text <-> Image search via shared CLIP embedding space"))

    # Stage 0: Create demo images
    images = create_demo_images()

    if not images:
        console.print("[red]Could not create images. Exiting.[/red]")
        return

    # Prepare some text chunks that relate to the images
    text_chunks = [
        "A vector database stores high-dimensional numerical representations (embeddings) "
        "of documents, images, or other data. It enables fast similarity search using "
        "algorithms like cosine similarity or dot product. Examples include Chroma, "
        "Pinecone, Weaviate, and Milvus.",
        "BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks documents "
        "based on query term frequencies. It is the foundation of many traditional search "
        "engines including Elasticsearch and Lucene. BM25 uses TF-IDF weighting with "
        "document length normalization.",
        "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for "
        "approximate nearest neighbor (ANN) search. It builds a multi-layer graph where "
        "upper layers contain fewer nodes for fast long-distance navigation, and lower "
        "layers provide fine-grained local search. It achieves sub-linear query complexity.",
        "RAG (Retrieval-Augmented Generation) combines information retrieval with LLM "
        "generation. The system retrieves relevant documents from a knowledge base and "
        "uses them as context for the language model to generate grounded answers.",
    ]

    # Stage 1: Try to load CLIP
    clip_model = load_clip_model()

    if clip_model is None:
        # FALLBACK: Text-only demo
        text_only_fallback(images)
        console.print("\n[bold green]Done.[/bold green] Multi-Modal RAG demo complete (fallback mode).\n")
        return

    # Stage 2: Build unified vector store
    vector_store = build_multimodal_store(clip_model, images, text_chunks)

    # Stage 3a: Text -> Image query
    console.print("\n" + "=" * 60 + "\n")
    cross_modal_query_text_to_image(
        clip_model, vector_store,
        "a diagram about vector databases and embeddings"
    )

    # Stage 3b: Image -> Text query
    console.print("\n" + "=" * 60 + "\n")
    # Use the HNSW Graph image as the query
    hnsw_image = images[2]  # Third image is "HNSW Graph"
    cross_modal_query_image_to_text(clip_model, vector_store, hnsw_image)

    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel("""
[bold]Multi-Modal RAG Architecture Summary:[/bold]

  Images + Text --> CLIP Encoder --> Shared 512-dim Space --> Chroma Store
                                                          |
                          Text Query <-------------------+
                          Image Query <------------------+
                                                          |
                                                    Retrieve Top-K
                                                          |
                                                    LLM Generation
    """, title="[green]Pipeline Complete[/green]", border_style="green"))

    console.print("\n[bold green]Done.[/bold green] Multi-Modal RAG pipeline complete.\n")


if __name__ == "__main__":
    main()
