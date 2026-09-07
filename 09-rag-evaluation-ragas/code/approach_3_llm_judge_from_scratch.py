"""Approach 3 — DIY LLM-as-judge from scratch (no RAGAS).

A structured rubric prompt asks an LLM to score a Q / context / answer triple
on three criteria, each 1-5:

  • faithfulness   — is every statement in the answer supported by the context?
  • relevance      — does the answer address the question directly?
  • completeness   — does the answer cover what the reference answer covers?

The judge must reply with STRICT JSON. Parsing is defensive: markdown fences
are stripped, the outermost {...} block is extracted, keys are validated, and
malformed output triggers a retry (up to 3 attempts) with the parse error fed
back into the prompt. Mean scores per criterion are aggregated at the end.

Evaluates 3 sample triples from the golden dataset (data/eval_dataset.json),
using the reference answer as the "generated" answer so the run is
deterministic and needs no separate generator call.

Run from the project root:
    uv run python 09-rag-evaluation-ragas/code/approach_3_llm_judge_from_scratch.py

Needs an LLM (Yolo-Auto cloud via YOLO_AUTO_API_KEY, or OpenAI/Ollama).
Degrades gracefully with a hint if none is reachable.
"""
import sys, pathlib, json, re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- Course convention: make the project root importable --------------------
MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]     # 09-rag-evaluation-ragas
PROJECT_ROOT = MODULE_DIR.parent                             # advance-rag
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))  # sibling scripts

from shared.config import get_llm, get_llm_provider_name  # noqa: E402
from rich.console import Console          # noqa: E402
from rich.table import Table              # noqa: E402
from rich.panel import Panel              # noqa: E402
from rich import box                      # noqa: E402

console = Console()

DATASET_PATH = MODULE_DIR / "data" / "eval_dataset.json"
N_EXAMPLES = 3
MAX_RETRIES = 3
CRITERIA = ["faithfulness", "relevance", "completeness"]

RUBRIC_PROMPT = """You are a strict RAG answer-quality judge. Score the ANSWER below on exactly three criteria, each an integer from 1 to 5.

Criteria:
- faithfulness: 5 = every statement is fully supported by the CONTEXT; 1 = mostly unsupported/hallucinated.
- relevance: 5 = the answer directly addresses the QUESTION with no filler; 1 = off-topic or evasive.
- completeness: 5 = the answer covers everything the REFERENCE ANSWER covers; 1 = almost nothing covered.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

REFERENCE ANSWER:
{reference}
{feedback}
Respond with ONLY a JSON object — no markdown, no commentary — in exactly this shape:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>, "completeness": <int 1-5>, "reasoning": "<one short sentence>"}}"""


# --------------------------------------------------------------------------- #
# Strict JSON parsing with retry-on-malformed                                 #
# --------------------------------------------------------------------------- #
def parse_judge_json(raw: str) -> dict:
    """Parse the judge's reply; raise ValueError on anything malformed."""
    text = raw.strip()
    # Strip markdown code fences if the model added them anyway.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Extract the outermost JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object found in reply: {raw[:120]!r}")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"top-level JSON is not an object: {type(obj).__name__}")
    scores = {}
    for key in CRITERIA:
        val = obj.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not (1 <= float(val) <= 5):
            raise ValueError(f"criterion '{key}' missing or out of range 1-5: {val!r}")
        scores[key] = int(round(float(val)))
    return {"scores": scores, "reasoning": str(obj.get("reasoning", ""))}


def judge_once(llm, question: str, context: str, answer: str, reference: str) -> dict:
    """One judged triple, with up to MAX_RETRIES attempts on malformed output."""
    feedback = ""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        prompt = RUBRIC_PROMPT.format(question=question, context=context,
                                      answer=answer, reference=reference,
                                      feedback=feedback)
        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        try:
            parsed = parse_judge_json(raw)
            if attempt > 1:
                console.print(f"   [dim]recovered on attempt {attempt}[/dim]")
            return parsed
        except Exception as e:
            last_err = e
            console.print(f"   [yellow]attempt {attempt}: malformed judge output ({e})[/yellow]")
            feedback = (f"\nYour previous reply was NOT valid JSON ({e}). "
                        f"Reply again with ONLY the JSON object.")
    raise RuntimeError(f"judge failed after {MAX_RETRIES} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    console.print(Panel.fit(
        "[bold]DIY LLM-as-judge — structured rubric, strict JSON, retry-on-malformed[/bold]\n"
        f"Judge: [cyan]{get_llm_provider_name()}[/cyan] · "
        f"{N_EXAMPLES} triples · max {MAX_RETRIES} retries per triple",
        title="⚖️ Approach 3: LLM Judge from Scratch", border_style="magenta"))

    try:
        llm = get_llm(temperature=0.0)
    except Exception as e:
        console.print(Panel(
            f"[red]Could not create an LLM:[/red] {e}\n\n"
            "Fix: set YOLO_AUTO_API_KEY (or OPENAI_API_KEY) in .env, or start Ollama.\n"
            "This script needs an LLM for judging.",
            title="Graceful stop", border_style="red"))
        sys.exit(0)

    if not DATASET_PATH.exists():
        console.print(Panel(
            f"[red]Golden dataset not found:[/red] {DATASET_PATH}\n\n"
            "Build it first:\n  uv run python 09-rag-evaluation-ragas/code/build_dataset.py",
            title="Missing dataset", border_style="red"))
        sys.exit(0)
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))[:N_EXAMPLES]

    results = []
    for i, ex in enumerate(dataset, 1):
        context = "\n\n".join(ex["contexts"])
        console.print(f"\n[bold cyan]{i}.[/bold cyan] {ex['question']}")
        try:
            parsed = judge_once(llm, ex["question"], context,
                                ex["ground_truth"], ex["ground_truth"])
        except Exception as e:
            console.print(f"   [red]skipped: {e}[/red]")
            continue
        s = parsed["scores"]
        results.append({"question": ex["question"], **s})
        console.print(f"   faithfulness={s['faithfulness']}  relevance={s['relevance']}  "
                      f"completeness={s['completeness']}")
        console.print(f"   [dim]{parsed['reasoning']}[/dim]")

    if not results:
        console.print(Panel(
            "[red]No triples could be judged.[/red]\n"
            "Check your API key / network, then re-run.",
            title="Graceful stop", border_style="red"))
        sys.exit(0)

    t = Table(title="Per-triple judge scores (1-5)", box=box.SIMPLE_HEAVY)
    t.add_column("#", justify="right", style="cyan")
    t.add_column("Question", max_width=44)
    for c in CRITERIA:
        t.add_column(c, justify="right")
    for i, r in enumerate(results, 1):
        t.add_row(str(i), r["question"][:44], *[str(r[c]) for c in CRITERIA])
    console.print(t)

    m = Table(title="Mean judge score per criterion (higher is better)", box=box.ROUNDED)
    m.add_column("Criterion", style="bold")
    m.add_column("Mean", justify="right")
    m.add_column("Range", justify="right")
    for c in CRITERIA:
        vals = [r[c] for r in results]
        m.add_row(c, f"{sum(vals) / len(vals):.2f}", f"{min(vals)}–{max(vals)}")
    console.print(m)
    console.print("\n[dim]Known LLM-judge biases: self-preference, verbosity bias, "
                  "non-determinism — use a different judge model than the generator "
                  "and spot-audit a sample by hand.[/dim]")


if __name__ == "__main__":
    main()
