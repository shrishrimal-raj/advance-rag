import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The Document Intelligence Pipeline (Week 9 weekly build).

Stages: extract (OCR/vision - simulated here) -> clean -> understand/structure
(one cloud LLM call emits JSON fields). DRY-RUN by default (prints stages, no model).
--selftest runs the full chain on a sample invoice with 1 LLM call. --run --image PATH
would invoke a real OCR/vision model (not on this machine).
"""
import argparse
import json
import pathlib
import re
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from shared.config import get_llm  # noqa: E402

# Simulated OCR output of an invoice (where a real vision/OCR model plugs in).
SAMPLE_EXTRACTED = """INVOICE #A-1042
Vendor: Acme Widgets Ltd
Date: 2026-08-14

Line items:
1x Widget Pro      $120.00
2x Widget Basic    $40.00
Shipping             $12.50

Total: $172.50
Due: 2026-09-13
"""


def stage_extract(image_path=None):
    """Real impl: OCR/vision model. Here we simulate OCR output."""
    if image_path:
        return f"(would OCR {image_path})\n" + SAMPLE_EXTRACTED
    return SAMPLE_EXTRACTED


def stage_clean(text):
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def stage_structure(clean_text):
    llm = get_llm()
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = llm.invoke([
        SystemMessage("You extract invoice fields. Respond with ONLY JSON: {vendor, date, line_items:[{desc,amount}], total, due}."),
        HumanMessage(clean_text),
    ])
    txt = resp.content.strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {"raw": txt}
    except Exception:
        return {"raw": txt}


def dry_run():
    print("=== Document Intelligence Pipeline (dry-run) ===")
    print("stages:")
    print("  1. extract  : OCR / vision model pulls text + layout  [simulated here]")
    print("  2. clean    : normalize whitespace, drop blank lines")
    print("  3. structure: LLM extracts JSON fields (vendor, line_items, total, due)")
    print("\n[dry-run] no model loaded. Use --selftest (1 LLM call) or --run --image PATH.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    if not (args.selftest or args.run):
        dry_run()
        return 0
    print("=== Document Intelligence Pipeline ===")
    raw = stage_extract(args.image)
    clean = stage_clean(raw)
    print("[extract+clean] ok, chars:", len(clean))
    try:
        structured = stage_structure(clean)
    except Exception as e:
        print(f"[structure] unavailable ({type(e).__name__}: {e})")
        print("Hint: needs a configured LLM (Yolo-Auto). Extract/clean run offline.")
        return 0
    print("\nSTRUCTURED:")
    print(json.dumps(structured, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
