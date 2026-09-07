"""PII guardrails & access control for RAG (Module 11) — fully offline.

Three production concerns, demonstrated end-to-end:

1. PII DETECTION   — regex detectors for email, phone, SSN, credit card (Luhn-checked).
2. PII REDACTION   — applied PRE-INDEXING so personal data never enters the vector store
                     (data minimization: don't store what you can't be asked to delete).
3. ACCESS CONTROL  — metadata ACL filtering: every chunk carries `classification`
                     (public / internal / restricted); retrieval filters by the caller's
                     clearance level, so a 'public' user can never retrieve 'restricted' docs.

Plus a GDPR / HIPAA-relevant compliance checklist printed at the end.

Run:  uv run python 11-production-rag/code/security_compliance.py
"""
from __future__ import annotations

import re
import sys
import pathlib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))  # -> project root

from rich.console import Console
from rich.table import Table

console = Console()

# --------------------------------------------------------------------------- #
# 1. PII detectors (regex)
# --------------------------------------------------------------------------- #
# Single-pass detector: ordered alternatives so SSN/CC win over the looser phone
# pattern (otherwise "123-45-6789" matches as a phone AND an SSN).
MASTER_PII_RE = re.compile(
    r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"|(?P<ssn>(?<!\d)\d{3}-\d{2}-\d{4}(?!\d))"
    r"|(?P<credit_card>(?<!\d)(?:\d[ \-]?){13,19}(?!\d))"
    r"|(?P<phone>(?:(?:\+|00)\d{1,3}[\s.\-]?)?(?:\(\d{1,4}\)[\s.\-]?)?"
    r"\d{2,4}[\s.\-]?\d{2,4}[\s.\-]?\d{2,4}(?!\d))"
)

PLACEHOLDERS = {
    "email": "[REDACTED_EMAIL]",
    "ssn": "[REDACTED_SSN]",
    "credit_card": "[REDACTED_CC]",
    "phone": "[REDACTED_PHONE]",
}


def luhn_valid(digits: str) -> bool:
    """Luhn checksum — filters out random digit runs that match the CC shape."""
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _classify(m: re.Match) -> str | None:
    """Map a MASTER_PII_RE match to a PII type (None = not actually PII)."""
    kind = m.lastgroup
    if kind == "credit_card" and not luhn_valid(m.group(0).replace(" ", "").replace("-", "")):
        return None  # digit run fails Luhn → not a card number
    return kind


def detect_pii(text: str) -> list[dict]:
    """Return [{type, value, span}] for every PII hit (single pass, no double-counting)."""
    found = []
    for m in MASTER_PII_RE.finditer(text):
        kind = _classify(m)
        if kind:
            found.append({"type": kind, "value": m.group(0), "span": m.span()})
    return found


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace each PII occurrence with its placeholder. Returns (clean_text, counts)."""
    counts: dict[str, int] = {}

    def _sub(m: re.Match) -> str:
        kind = _classify(m)
        if kind is None:
            return m.group(0)
        counts[kind] = counts.get(kind, 0) + 1
        return PLACEHOLDERS[kind]

    return MASTER_PII_RE.sub(_sub, text), counts


# --------------------------------------------------------------------------- #
# 2. Pre-indexing redaction demo
# --------------------------------------------------------------------------- #
RAW_DOCUMENTS = [
    {
        "id": "doc-1",
        "classification": "public",
        "text": "Acme Robotics' Falcon AMR uses LiDAR + SLAM navigation. Support: help@acme-robotics.com.",
    },
    {
        "id": "doc-2",
        "classification": "internal",
        "text": "Q3 incident: warehouse bot jammed. Escalate to Priya (priya.sharma@acme-robotics.com, "
                "+1 (415) 555-0132). Internal review only.",
    },
    {
        "id": "doc-3",
        "classification": "restricted",
        "text": "Customer refund case: John Doe, SSN 123-45-6789, card 4111 1111 1111 1111. "
                "Do not share outside finance.",
    },
]


def build_index(docs: list[dict]) -> list[dict]:
    """Simulated ingestion: redact PII BEFORE storing (data minimization)."""
    index = []
    for d in docs:
        clean, counts = redact(d["text"])
        index.append({**d, "stored_text": clean, "redacted_counts": counts})
    return index


# --------------------------------------------------------------------------- #
# 3. Access control: metadata ACL filtering
# --------------------------------------------------------------------------- #
CLEARANCE_ORDER = {"public": 0, "internal": 1, "restricted": 2}


def acl_filter(index: list[dict], clearance: str) -> list[dict]:
    """A caller may only see chunks classified AT OR BELOW their clearance level."""
    level = CLEARANCE_ORDER.get(clearance, -1)
    return [c for c in index if CLEARANCE_ORDER[c["classification"]] <= level]


def simulate_retrieval(index: list[dict], clearance: str, query_terms: set[str]) -> list[dict]:
    """ACL filter FIRST, then naive keyword relevance — order matters (no leakage via ranking)."""
    visible = acl_filter(index, clearance)
    scored = [(sum(1 for t in query_terms if t in c["stored_text"].lower()), c) for c in visible]
    scored.sort(key=lambda sc: sc[0], reverse=True)
    return [c for score, c in scored if score > 0][:3]


# --------------------------------------------------------------------------- #
# Compliance checklist
# --------------------------------------------------------------------------- #
CHECKLIST = [
    ("GDPR", "Data minimization", "Redact PII pre-indexing; store only what answers require", "✔ done in this demo"),
    ("GDPR", "Right to erasure", "Delete a person's data by metadata query (name/email hash)", "needs per-person metadata keys"),
    ("GDPR", "Purpose limitation", "Document why each data category is indexed; review on change", "process, not code"),
    ("GDPR", "Portability", "Export a user's stored chunks on request", "needs export endpoint"),
    ("HIPAA", "Minimum necessary", "Role-based ACL so staff see only what their role needs", "✔ done via classification filter"),
    ("HIPAA", "Audit logging", "Log who queried what, which chunks were returned, with request ID", "wire into monitoring.py"),
    ("HIPAA", "BAA / vendor review", "If the LLM provider sees PHI, a Business Associate Agreement is required", "check Yolo-Auto/OpenAI terms"),
    ("Both", "Encryption", "Encrypt vector store at rest and TLS in transit", "infra config"),
]


def main() -> None:
    console.rule("🔐 Security & Compliance Demo")

    # -- Detection + redaction ------------------------------------------------ #
    console.print("[bold]1. PII detection & pre-indexing redaction[/bold]\n")
    t1 = Table(title="Detected PII in raw documents")
    t1.add_column("Doc")
    t1.add_column("Type")
    t1.add_column("Value")
    for d in RAW_DOCUMENTS:
        for hit in detect_pii(d["text"]):
            t1.add_row(d["id"], hit["type"], hit["value"])
    console.print(t1)

    index = build_index(RAW_DOCUMENTS)
    t2 = Table(title="Stored (redacted) text — what actually enters the vector DB")
    t2.add_column("Doc")
    t2.add_column("Classification")
    t2.add_column("Stored text", max_width=70)
    t2.add_column("Redacted")
    for c in index:
        n = sum(c["redacted_counts"].values())
        t2.add_row(c["id"], c["classification"], c["stored_text"],
                   f"{n} value(s)" if n else "[green]clean[/green]")
    console.print(t2)

    # -- Access control -------------------------------------------------------- #
    console.print("\n[bold]2. Metadata ACL filtering (retrieval respects clearance)[/bold]\n")
    query_terms = {"acme", "support"}
    t3 = Table(title=f"Retrieval for terms {sorted(query_terms)} under different clearances")
    t3.add_column("Caller clearance")
    t3.add_column("Chunks visible")
    t3.add_column("Top results returned")
    for clearance in ("public", "internal", "restricted"):
        results = simulate_retrieval(index, clearance, query_terms)
        visible_ids = [c["id"] for c in acl_filter(index, clearance)]
        t3.add_row(clearance, ", ".join(visible_ids),
                   ", ".join(f"{r['id']} ({r['classification']})" for r in results) or "—")
    console.print(t3)
    console.print("[dim]Note: the 'public' caller never even SEES doc-2/doc-3 — the filter runs before ranking.[/dim]")

    # -- Compliance checklist ---------------------------------------------------- #
    console.print("\n[bold]3. Compliance checklist (GDPR / HIPAA-relevant)[/bold]\n")
    t4 = Table(title="Compliance checklist")
    t4.add_column("Regime")
    t4.add_column("Requirement")
    t4.add_column("What it means here", max_width=55)
    t4.add_column("Status")
    for regime, req, meaning, status in CHECKLIST:
        t4.add_row(regime, req, meaning, status)
    console.print(t4)

    console.print("\n[green]✔ PII never reaches the index; retrieval is clearance-scoped; "
                  "checklist shows what's left to wire up.[/green]")


if __name__ == "__main__":
    main()
