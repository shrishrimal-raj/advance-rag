import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audited_tool_bridge.core import TokenBucket, ToolBridge  # noqa: E402


def test_auth_rejects_invalid_key():
    b = ToolBridge({"k1": "alice"})
    b.register_tool("add", lambda a, b: a + b)
    res = b.invoke("bad", "add", {"a": 1, "b": 2})
    assert res == {"ok": False, "error": "unauthorized"}
    assert b.audit[-1].allowed is False and b.audit[-1].reason == "unauthorized"


def test_valid_call_succeeds_and_audits():
    b = ToolBridge({"k1": "alice"})
    b.register_tool("add", lambda a, b: a + b)
    res = b.invoke("k1", "add", {"a": 2, "b": 3})
    assert res == {"ok": True, "result": 5}
    assert b.audit[-1].allowed is True and b.audit[-1].reason == "ok"


def test_rate_limit_enforced():
    b = ToolBridge({"k1": "alice"}, rate_capacity=2, rate_refill_per_sec=0.0)
    b.register_tool("noop", lambda: "x")
    r = [b.invoke("k1", "noop", {}) for _ in range(4)]
    assert sum(1 for x in r if x["ok"]) == 2
    assert any(x.get("error") == "rate_limited" for x in r)


def test_unknown_tool_audited():
    b = ToolBridge({"k1": "alice"})
    res = b.invoke("k1", "nope", {})
    assert res["error"].startswith("unknown_tool")
    assert b.audit[-1].reason == "unknown_tool"


def test_audit_filter_by_client():
    b = ToolBridge({"k1": "alice", "k2": "bob"})
    b.register_tool("t", lambda: 1)
    b.invoke("k1", "t", {})
    b.invoke("k2", "t", {})
    alice = b.audit_log("alice")
    assert len(alice) == 1 and all(e["client"] == "alice" for e in alice)


def test_token_bucket_refills():
    t = {"now": 0.0}
    tb = TokenBucket(capacity=1, refill_per_sec=100.0, clock=lambda: t["now"])
    assert tb.allow() is True   # 1 -> 0
    assert tb.allow() is False  # same instant, no refill yet
    t["now"] += 0.02            # advance 20ms -> refills past 1 (capped at capacity)
    assert tb.allow() is True
