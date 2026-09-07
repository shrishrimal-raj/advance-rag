"""LangGraph plan-and-execute + reflection research agent.

Topology:

    START -> planner -> executor -> reflector --(confident)--> synthesizer -> END
                                  ^            |
                                  |            +(low confidence, first time)
                                  +------------+   (append revision steps)

The reflector gets exactly ONE revision pass (guarded by `revision_done`).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterator

from langgraph.graph import END, START, StateGraph

from .prompts import PLANNER_PROMPT, REFLECT_PROMPT, SYNTH_PROMPT
from .state import AgentState
from .tools import ResearchTools, get_tools

CONFIDENCE_THRESHOLD = int(os.getenv("REFLECT_CONFIDENCE_THRESHOLD", "60"))


# --- citation helpers -------------------------------------------------------

def extract_citation_indices(answer: str) -> list[int]:
    """All [n] markers in the answer, in order of appearance."""
    return [int(m) for m in re.findall(r"\[(\d+)\]", answer)]


def sanitize_citations(answer: str, num_sources: int) -> str:
    """Drop [n] markers that reference non-existent sources."""
    def _keep(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= num_sources else ""

    return re.sub(r"\[(\d+)\]", _keep, answer).strip()


def build_citations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Number unique sources across all observations, in first-seen order."""
    seen: dict[str, dict[str, Any]] = {}
    for obs in observations:
        for src in obs.get("sources", []):
            name = src if isinstance(src, str) else str(src.get("source", "unknown"))
            if name not in seen:
                seen[name] = {"index": len(seen) + 1, "source": name}
    return list(seen.values())


def format_numbered_sources(citations: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{c['index']}] {c['source']}" for c in citations)


# --- JSON extraction (LLM output is not always clean) ------------------------

def _extract_json(text: str) -> Any:
    text = text.strip()
    # strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _observations_summary(observations: list[dict[str, Any]], max_chars: int = 3000) -> str:
    parts = []
    for i, obs in enumerate(observations, 1):
        result = obs.get("result")
        if isinstance(result, list):
            snippet = "\n".join(f"- ({r.get('source')}) {r.get('text', '')[:220]}" for r in result[:4])
        else:
            snippet = str(result)[:220]
        parts.append(f"{i}. [{obs.get('tool', '?')}] {snippet}")
    summary = "\n".join(parts)
    return summary[:max_chars]


# --- graph construction ------------------------------------------------------

def build_nodes(llm: Any, tools: ResearchTools) -> dict[str, Any]:
    """Node functions closing over the injected llm and tools (testable)."""

    def planner(state: AgentState) -> dict[str, Any]:
        raw = llm.invoke(PLANNER_PROMPT.format(question=state["question"])).content
        data = _extract_json(raw)
        steps: list[dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("tool") and item.get("input"):
                    steps.append({"tool": item["tool"], "input": str(item["input"]), "purpose": str(item.get("purpose", ""))})
        if not steps:  # graceful fallback: one broad search
            steps = [{"tool": "vector_search", "input": state["question"], "purpose": "broad semantic search"}]
        return {"plan": steps}

    def executor(state: AgentState) -> dict[str, Any]:
        executed = len(state.get("steps", []))
        new_steps, new_obs = [], []
        for step in state.get("plan", [])[executed:]:
            tool_name, input_ = step["tool"], step["input"]
            try:
                result = tools.run(tool_name, input_)
            except Exception as exc:  # noqa: BLE001 - a bad tool call must not kill the run
                result = f"error: {exc}"
            sources = [r["source"] for r in result] if isinstance(result, list) else []
            new_steps.append({**step, "status": "ok" if not str(result).startswith("error") else "error"})
            new_obs.append({"step": input_, "tool": tool_name, "result": result, "sources": sources})
        return {"steps": state.get("steps", []) + new_steps, "observations": state.get("observations", []) + new_obs}

    def reflector(state: AgentState) -> dict[str, Any]:
        raw = llm.invoke(REFLECT_PROMPT.format(question=state["question"], observations=_observations_summary(state.get("observations", [])))).content
        data = _extract_json(raw)
        confidence, critique, missing = 0, "", []
        if isinstance(data, dict):
            try:
                confidence = int(data.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0
            critique = str(data.get("critique", ""))
            missing = [str(x) for x in data.get("missing", [])][:3]
        reflection = {"confidence": confidence, "critique": critique, "missing": missing}
        update: dict[str, Any] = {"reflections": state.get("reflections", []) + [reflection]}
        if confidence < CONFIDENCE_THRESHOLD and not state.get("revision_done"):
            terms = missing or [state["question"]]
            update["plan"] = state.get("plan", []) + [
                {"tool": "vector_search", "input": t, "purpose": f"revision: fill gap ({critique[:80]})"} for t in terms
            ]
            update["revision_done"] = True
        return update

    def synthesizer(state: AgentState) -> dict[str, Any]:
        citations = build_citations(state.get("observations", []))
        raw = llm.invoke(SYNTH_PROMPT.format(question=state["question"], sources=format_numbered_sources(citations))).content
        answer = sanitize_citations(raw.strip(), len(citations))
        return {"answer": answer, "citations": citations}

    return {"planner": planner, "executor": executor, "reflector": reflector, "synthesizer": synthesizer}


def build_graph(llm: Any, tools: ResearchTools):
    """Compile the LangGraph with conditional edges."""
    nodes = build_nodes(llm, tools)
    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "reflector")

    def route_after_reflection(state: AgentState) -> str:
        last = state.get("reflections", [{}])[-1]
        if last.get("confidence", 0) < CONFIDENCE_THRESHOLD and not state.get("revision_done"):
            return "executor"  # one revision pass
        return "synthesizer"

    g.add_conditional_edges("reflector", route_after_reflection, {"executor": "executor", "synthesizer": "synthesizer"})
    g.add_edge("synthesizer", END)
    return g.compile()


# --- public entrypoints -------------------------------------------------------

def run_research(question: str, llm: Any = None, tools: ResearchTools | None = None) -> dict[str, Any]:
    """Run the full graph; returns final state (answer, citations, plan, ...)."""
    llm = llm or _default_llm()
    tools = tools or get_tools()
    graph = build_graph(llm, tools)
    return graph.invoke({"question": question})


def stream_research(question: str, llm: Any = None, tools: ResearchTools | None = None) -> Iterator[tuple[str, Any]]:
    """Run planning/execution/reflection, then stream synthesis tokens.

    Yields ("phase", label) events, then ("token", text) events, then
    ("done", final_state_dict).
    """
    llm = llm or _default_llm()
    tools = tools or get_tools()
    nodes = build_nodes(llm, tools)
    state: AgentState = {"question": question}
    yield ("phase", "planning")
    state.update(nodes["planner"](state))
    yield ("phase", "executing")
    state.update(nodes["executor"](state))
    yield ("phase", "reflecting")
    state.update(nodes["reflector"](state))
    if state.get("reflections", [{}])[-1].get("confidence", 0) < CONFIDENCE_THRESHOLD and not state.get("revision_done"):
        state.update(nodes["executor"](state))
        state.update(nodes["reflector"](state))
    citations = build_citations(state.get("observations", []))
    prompt = SYNTH_PROMPT.format(question=question, sources=format_numbered_sources(citations))
    yield ("phase", "synthesizing")
    chunks: list[str] = []
    for chunk in llm.stream(prompt):
        piece = chunk.content if hasattr(chunk, "content") else str(chunk)
        if piece:
            chunks.append(piece)
            yield ("token", piece)
    answer = sanitize_citations("".join(chunks).strip(), len(citations))
    state.update({"answer": answer, "citations": citations})
    yield ("done", state)


_llm_instance: Any = None


def _default_llm() -> Any:
    """3-tier LLM selection mirroring shared/config.py (OpenAI -> Yolo-Auto -> Ollama)."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    import os as _os

    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    openai_key = _os.getenv("OPENAI_API_KEY", "").strip()
    yolo_key = _os.getenv("YOLO_AUTO_API_KEY", "").strip()
    if openai_key:
        from langchain_openai import ChatOpenAI

        _llm_instance = ChatOpenAI(model=_os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)
    elif yolo_key:
        from langchain_openai import ChatOpenAI

        _llm_instance = ChatOpenAI(
            model=_os.getenv("YOLO_AUTO_MODEL", "qwen3.8-27b"),
            base_url=_os.getenv("YOLO_AUTO_BASE_URL", "https://yolo-auto.com/v1"),
            api_key=yolo_key,
            temperature=0.0,
            max_retries=2,
        )
    else:
        from langchain_ollama import ChatOllama

        _llm_instance = ChatOllama(model=_os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=0.0)
    return _llm_instance
