"""Prompt templates for the agentic research graph.

Each prompt carries a machine-readable marker on its first line (PLAN:, REFLECT:,
SYNTH:) so tests can route canned responses deterministically, and so logs are
easy to grep in production.
"""
from __future__ import annotations

PLANNER_PROMPT = """PLAN: You are a research planner. Break the question into 2-4 concrete research steps.
Available tools:
- vector_search: {"tool": "vector_search", "input": "<query string>", "purpose": "..."} — semantic search over the local document corpus
- metadata_filter_search: {"tool": "metadata_filter_search", "input": "<field>=<value>", "purpose": "..."} — exact metadata filter, e.g. "topic=chunking" or "source=<doc title>"
- calculator: {"tool": "calculator", "input": "<python arithmetic expression>", "purpose": "..."} — numeric computation

Rules:
- Every step must be one of the three tools above.
- Prefer vector_search for conceptual questions, metadata_filter_search when the question names a specific doc/topic, calculator only for numbers.
- Respond with ONLY a JSON array of step objects. No prose, no markdown fences.

Question: {question}
"""

REFLECT_PROMPT = """REFLECT: You are a research critic. Judge whether the gathered observations are sufficient to answer the question well.

Question: {question}

Observations so far:
{observations}

Respond with ONLY a JSON object:
{{"confidence": <integer 0-100>, "critique": "<one sentence>", "missing": ["<term>", ...]}}
- confidence >= 60 means the evidence is sufficient.
- If insufficient, list 1-3 specific missing terms that a follow-up search should target.
"""

SYNTH_PROMPT = """SYNTH: You are a research synthesizer. Write a clear, well-structured answer to the question using ONLY the numbered sources below.

Question: {question}

Numbered sources:
{sources}

Citation rules (strict):
- Cite every factual claim inline as [n] where n is the source number above.
- Use only source numbers that exist in the list. No invented sources.
- End with a "Sources:" section listing each used number and its title.

Write the answer now.
"""
