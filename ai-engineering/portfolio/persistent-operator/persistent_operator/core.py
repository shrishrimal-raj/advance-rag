"""The Persistent Operator core.

A **stateful tool-calling agent** whose conversation and tool-call history are
persisted to an external store (Redis in production, in-memory for tests) so it
survives process restarts. All state lives under `store` keyed by session_id —
never only in memory.
"""
import json
from typing import Callable, Dict, List, Optional


class MemoryStore:
    """In-memory stand-in for Redis (string get/set by key)."""

    def __init__(self):
        self._d: Dict[str, str] = {}

    def get(self, key: str):
        return self._d.get(key)

    def set(self, key: str, value: str):
        self._d[key] = value

    def delete(self, key: str):
        self._d.pop(key, None)


class PersistentOperator:
    """Stateful agent. State is loaded from / saved to `store` per operation."""

    def __init__(self, store, session_id: str = "default",
                 tools: Optional[Dict[str, Callable]] = None):
        self.store = store
        self.session_id = session_id
        self.tools = tools or {}
        self._state = self._load()

    def _key(self) -> str:
        return f"operator:{self.session_id}"

    def _load(self) -> dict:
        raw = self.store.get(self._key())
        if not raw:
            return {"messages": [], "tool_calls": []}
        return json.loads(raw)

    def _save(self):
        self.store.set(self._key(), json.dumps(self._state))

    @property
    def messages(self) -> List[dict]:
        return list(self._state["messages"])

    @property
    def tool_calls(self) -> List[dict]:
        return list(self._state["tool_calls"])

    def add_message(self, role: str, content: str):
        self._state["messages"].append({"role": role, "content": content})
        self._save()

    def call_tool(self, name: str, args: dict) -> str:
        """Invoke a registered tool, record the call + result, persist."""
        fn = self.tools.get(name)
        if fn is None:
            result = f"error: unknown tool '{name}'"
        else:
            try:
                result = str(fn(**args))
            except Exception as e:  # noqa: BLE001
                result = f"error: {e}"
        self._state["tool_calls"].append({"name": name, "args": args, "result": result})
        self._state["messages"].append({"role": "tool", "content": f"{name} -> {result}"})
        self._save()
        return result

    def step(self, user_input: str, decide: Callable[[dict], dict]) -> dict:
        """One agent turn: record the user message, let `decide(state)` pick a tool
        or produce an answer, then persist. Returns the turn outcome.

        decide returns {"tool": name, "args": {...}} or {"answer": "..."}.
        """
        self.add_message("user", user_input)
        action = decide(self._state)
        if "tool" in action:
            res = self.call_tool(action["tool"], action.get("args", {}))
            self.add_message("assistant", f"called {action['tool']}")
            return {"type": "tool", "result": res}
        ans = action.get("answer", "")
        self.add_message("assistant", ans)
        return {"type": "answer", "answer": ans}

    def history(self) -> dict:
        return {"messages": self.messages, "tool_calls": self.tool_calls}
