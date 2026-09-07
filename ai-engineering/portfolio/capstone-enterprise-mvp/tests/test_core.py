import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capstone.core import AgentGraph, build_capstone_graph  # noqa: E402


def test_run_produces_final():
    g = build_capstone_graph(lambda q: "retrieved snippet")
    st = g.run("what is X?")
    assert st.plan and st.context == "retrieved snippet"
    assert st.draft and st.final.endswith("(reviewed)")


def test_stream_yields_events():
    g = build_capstone_graph(lambda q: "ctx")
    evs = [e for e, _ in g.stream("q")]
    assert evs[0] == "planner:done"
    assert evs[-1] == "complete"
    assert "writer:done" in evs


def test_events_record_all_nodes():
    g = build_capstone_graph(lambda q: "c")
    st = g.run("q")
    names = [e.split(":")[0] for e in st.events]
    assert names == ["planner", "planner", "researcher", "researcher",
                     "writer", "writer", "critic", "critic"]


def test_custom_retriever_used():
    g = build_capstone_graph(lambda q: f"ctx:{q}")
    st = g.run("hello")
    assert st.context == "ctx:hello"
