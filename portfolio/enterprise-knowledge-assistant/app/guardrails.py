"""Guardrails: PII redaction + prompt-injection pattern filter.

Defense-in-depth for a public-facing RAG endpoint:
1. PII in user questions is redacted before retrieval/generation so personal
   data never reaches the LLM prompt or the logs.
2. Known prompt-injection phrasings block the request outright (400), because
   retrieved documents are untrusted content and we refuse to let users try to
   override the system prompt through the question channel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- PII patterns -----------------------------------------------------------
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("phone", re.compile(r"(?<!\d)(\+?\d[\d\s\-().]{7,18}\d)(?!\d)")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

# --- Prompt-injection patterns ----------------------------------------------
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_previous", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.I)),
    ("disregard", re.compile(r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions|prompts|rules)", re.I)),
    ("system_override", re.compile(r"(reveal|show|print|repeat)\s+(your|the)\s+(system\s+)?prompt", re.I)),
    ("role_hijack", re.compile(r"you\s+are\s+now\s+(?:a|an|in)\b", re.I)),
    ("jailbreak", re.compile(r"\b(jailbreak|DAN\s+mode|developer\s+mode|act\s+as\s+if\s+you\s+have\s+no\s+restrictions)\b", re.I)),
    ("instruction_inject", re.compile(r"new\s+instructions?:\s", re.I)),
]


@dataclass
class GuardrailResult:
    safe_question: str
    redactions: dict[str, int] = field(default_factory=dict)
    injection_hits: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.injection_hits)


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace PII spans with typed placeholders. Returns (clean_text, counts)."""
    counts: dict[str, int] = {}
    out = text
    for label, pattern in _PII_PATTERNS:
        out, n = pattern.subn(f"[REDACTED_{label.upper()}]", out)
        if n:
            counts[label] = n
    return out, counts


def detect_injection(text: str) -> list[str]:
    """Return the labels of all injection patterns matched."""
    return [label for label, pattern in INJECTION_PATTERNS if pattern.search(text)]


def sanitize_question(question: str) -> GuardrailResult:
    """Full guardrail pass: injection check first, then PII redaction."""
    hits = detect_injection(question)
    clean, counts = redact_pii(question)
    return GuardrailResult(safe_question=clean, redactions=counts, injection_hits=hits)
