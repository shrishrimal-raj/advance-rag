"""Typed state for the agentic research graph."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State flowing through planner -> executor -> reflector -> synthesizer.

    All fields are optional (total=False) so nodes can return partial updates;
    LangGraph merges them into the full state.
    """

    question: str                      # the user's research question
    plan: list[dict[str, Any]]         # planned steps: [{"tool": ..., "input": ..., "purpose": ...}]
    steps: list[dict[str, Any]]        # executed steps (mirrors plan entries + result summary)
    observations: list[dict[str, Any]] # tool results: [{"step": ..., "result": ..., "sources": [...]}]
    reflections: list[dict[str, Any]]  # critic outputs: [{"confidence": int, "critique": str, "missing": [...]}]
    revision_done: bool                # guard: at most ONE revision pass
    answer: str                        # final synthesized answer with [n] citations
    citations: list[dict[str, Any]]    # [{"index": 1, "source": "...", "title": "..."}]
