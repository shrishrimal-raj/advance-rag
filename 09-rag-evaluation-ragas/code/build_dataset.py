"""Build a small golden RAG evaluation dataset from data/samples/.

Each example = {question, ground_truth, contexts} where `contexts` are the
correct source chunk(s) pulled from the ACTUAL sample files (nothing fabricated).

Run from the project root:
    uv run python 09-rag-evaluation-ragas/code/build_dataset.py

Output: 09-rag-evaluation-ragas/data/eval_dataset.json
"""
import sys, pathlib, json

# --- Course convention: make the project root importable --------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm, get_embeddings  # noqa: F401  (course convention)

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]     # 09-rag-evaluation-ragas
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]   # advance-rag
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUT_PATH = MODULE_DIR / "data" / "eval_dataset.json"


def load_samples():
    """Read the four sample documents once."""
    profile = json.loads((SAMPLES_DIR / "company_profile.json").read_text(encoding="utf-8"))
    products_csv = (SAMPLES_DIR / "products.csv").read_text(encoding="utf-8").strip()
    rag_txt = (SAMPLES_DIR / "rag_overview.txt").read_text(encoding="utf-8").strip()
    vdb_md = (SAMPLES_DIR / "vector_db_notes.md").read_text(encoding="utf-8").strip()
    return profile, products_csv, rag_txt, vdb_md


def build_dataset():
    profile, products_csv, rag_txt, vdb_md = load_samples()

    # --- company_profile.json contexts (derived from real content) ----------
    falcon = next(p for p in profile["products"] if p["name"] == "Falcon AMR")
    heron = next(p for p in profile["products"] if p["name"] == "Heron Arm X2")
    support = profile["support_policy"]
    falcon_ctx = json.dumps(falcon, indent=2)
    heron_ctx = json.dumps(heron, indent=2)
    support_ctx = json.dumps(support, indent=2)

    # --- products.csv contexts (header + the specific row) ------------------
    csv_lines = [l for l in products_csv.splitlines() if l.strip()]
    header = csv_lines[0]

    def csv_ctx(product_name: str) -> str:
        row = next(l for l in csv_lines[1:] if l.split(",")[0] == product_name)
        return f"{header}\n{row}"

    # --- rag_overview.txt context (the specific paragraph) ------------------
    rag_paras = [p.strip() for p in rag_txt.split("\n\n") if p.strip()]
    rag_problems_para = next(p for p in rag_paras if p.startswith("RAG solves two major problems"))

    # --- vector_db_notes.md context (the HNSW section) ----------------------
    hnsw_section = next(s for s in vdb_md.split("### ") if s.startswith("HNSW"))
    hnsw_ctx = "### " + hnsw_section.strip()

    dataset = [
        {
            "question": "What is the maximum payload capacity of the Falcon AMR?",
            "ground_truth": "The Falcon AMR has a maximum payload capacity of 500 kg.",
            "contexts": [falcon_ctx],
        },
        {
            "question": "How many hours of battery life does the Falcon AMR provide?",
            "ground_truth": "The Falcon AMR provides 8 hours of battery life.",
            "contexts": [falcon_ctx],
        },
        {
            "question": "What warranty period and support response time does Acme Robotics offer?",
            "ground_truth": "Acme Robotics offers a 3-year warranty with a 4-hour support response time.",
            "contexts": [support_ctx],
        },
        {
            "question": "In which countries does Acme Robotics provide on-site support coverage?",
            "ground_truth": "Acme Robotics provides on-site support coverage in India, Singapore, and Dubai.",
            "contexts": [support_ctx],
        },
        {
            "question": "What is the repetitive accuracy of the Heron Arm X2 robotic arm?",
            "ground_truth": "The Heron Arm X2 has a repetitive accuracy of 0.05 mm.",
            "contexts": [heron_ctx],
        },
        {
            "question": "How much does the Quantum Laptop Pro 16 cost?",
            "ground_truth": "The Quantum Laptop Pro 16 costs $1,899.99.",
            "contexts": [csv_ctx("Quantum Laptop Pro 16")],
        },
        {
            "question": "What is the price and target region of the Atlas Standing Desk?",
            "ground_truth": "The Atlas Standing Desk is priced at $549.00 and targets the NAMER region.",
            "contexts": [csv_ctx("Atlas Standing Desk")],
        },
        {
            "question": "How many units of the Terra Desk Lamp LED are currently in stock?",
            "ground_truth": "There are 300 units of the Terra Desk Lamp LED in stock.",
            "contexts": [csv_ctx("Terra Desk Lamp LED")],
        },
        {
            "question": "What two major problems of large language models does RAG solve?",
            "ground_truth": ("RAG solves hallucination (by grounding answers in retrieved evidence) "
                             "and knowledge cutoff (by letting the knowledge base be updated "
                             "without retraining)."),
            "contexts": [rag_problems_para],
        },
        {
            "question": "What are the three key tunable parameters of the HNSW index and what does each control?",
            "ground_truth": ("HNSW uses M (max edges per node, default 16; higher improves recall but "
                             "uses more memory), efConstruction (build-time beam width; higher gives "
                             "better index quality but slower build), and efSearch (query-time beam "
                             "width; higher improves recall but slows queries)."),
            "contexts": [hnsw_ctx],
        },
    ]
    return dataset


def main():
    if not SAMPLES_DIR.exists():
        console.print(f"[red]Sample directory not found:[/red] {SAMPLES_DIR}")
        sys.exit(1)

    dataset = build_dataset()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    table = Table(title=f"Golden eval dataset ({len(dataset)} examples)", box=box.ROUNDED)
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Question", max_width=44)
    table.add_column("Ground truth", max_width=44)
    table.add_column("Contexts", justify="right")
    for i, ex in enumerate(dataset, 1):
        table.add_row(str(i), ex["question"], ex["ground_truth"], str(len(ex["contexts"])))
    console.print(table)
    console.print(f"\n[green]Saved[/green] {OUT_PATH}")


if __name__ == "__main__":
    main()
