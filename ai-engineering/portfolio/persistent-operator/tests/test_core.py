import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistent_operator.core import MemoryStore, PersistentOperator  # noqa: E402


def test_state_persists_across_restart():
    store = MemoryStore()
    tools = {"greet": lambda name: f"hi {name}"}
    op1 = PersistentOperator(store, "s1", tools)
    op1.add_message("user", "hello")
    op1.call_tool("greet", {"name": "raj"})
    # simulate restart: brand-new instance, same store + session id
    op2 = PersistentOperator(store, "s1", tools)
    assert any("greet" in m["content"] for m in op2.messages)
    assert op2.tool_calls[0]["name"] == "greet"
    assert op2.tool_calls[0]["result"] == "hi raj"


def test_sessions_are_isolated():
    store = MemoryStore()
    PersistentOperator(store, "a").add_message("user", "A")
    assert PersistentOperator(store, "b").messages == []


def test_unknown_tool_records_error():
    store = MemoryStore()
    op = PersistentOperator(store, "s", {})
    res = op.call_tool("nope", {})
    assert res.startswith("error:")
    assert op.tool_calls[-1]["name"] == "nope"


def test_step_tool_then_answer():
    store = MemoryStore()
    tools = {"add": lambda a, b: a + b}
    op = PersistentOperator(store, "s", tools)
    out = op.step("add 2+3", lambda st: {"tool": "add", "args": {"a": 2, "b": 3}})
    assert out == {"type": "tool", "result": "5"}
    out2 = op.step("done", lambda st: {"answer": "all set"})
    assert out2 == {"type": "answer", "answer": "all set"}
    roles = [m["role"] for m in op.messages]
    assert {"user", "assistant", "tool"} <= set(roles)


def test_history_shape():
    store = MemoryStore()
    op = PersistentOperator(store, "s", {"t": lambda: "x"})
    op.call_tool("t", {})
    h = op.history()
    assert set(h.keys()) == {"messages", "tool_calls"}
    assert len(h["tool_calls"]) == 1
