"""The Capstone core.

A **LangGraph-style multi-agent orchestration** that wires together a planner, a RAG
researcher, a writer, and a critic over a shared `State`, **streaming** intermediate
events for a live UI. Offline and testable via pluggable node callables; swap in real
LLM/RAG clients (and LangSmith tracing) in production.
"""
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class State:
    query: str
    plan: str = ""
    context: str = ""
    draft: str = ""
    final: str = ""
    events: List[str] = field(default_factory=list)


def planner_node(state: State) -> State:
    state.plan = f"answer: {state.query}"
    return state


def researcher_node(retriever: Callable[[str], str]) -> Callable[[State], State]:
    def _run(state: State) -> State:
        state.context = retriever(state.query)
        return state

    return _run


def writer_node(state: State) -> State:
    state.draft = f"[{state.plan}] using {state.context}"
    return state


def critic_node(state: State) -> State:
    state.final = state.draft + " (reviewed)"
    return state


class AgentGraph:
    """Runs an ordered list of named `(name, fn)` nodes over a shared State."""

    def __init__(self, nodes: List[tuple]):
        self.nodes = nodes

    def _step(self, state: State, name: str, fn: Callable[[State], State]) -> State:
        state.events.append(f"{name}:start")
        state = fn(state)
        state.events.append(f"{name}:done")
        return state

    def run(self, query: str) -> State:
        state = State(query=query)
        for name, fn in self.nodes:
            state = self._step(state, name, fn)
        return state

    def stream(self, query: str):
        """Yield `(event, state)` after each node for a streaming UI."""
        state = State(query=query)
        for name, fn in self.nodes:
            state = self._step(state, name, fn)
            yield state.events[-1], state
        yield "complete", state


def build_capstone_graph(retriever: Callable[[str], str]) -> AgentGraph:
    return AgentGraph([
        ("planner", planner_node),
        ("researcher", researcher_node(retriever)),
        ("writer", writer_node),
        ("critic", critic_node),
    ])
