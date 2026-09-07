import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dual_agent_supervisor.core import (  # noqa: E402
    CheckerVerdict,
    DualAgentSupervisor,
    ProviderOutage,
    make_failover,
)


def test_maker_checker_approves_first_try():
    maker = make_failover(lambda t: "good answer", lambda t: "good answer")
    checker = make_failover(lambda v: CheckerVerdict(True, "ok"), lambda v: CheckerVerdict(True, "ok"))
    sup = DualAgentSupervisor(maker, checker, max_retries=3)
    res = sup.run("task")
    assert res["ok"] is True and res["attempts"] == 1
    assert res["output"] == "good answer"


def test_retry_on_rejection_then_approve():
    calls = {"n": 0}

    def mk(t):
        calls["n"] += 1
        return "bad" if calls["n"] == 1 else "good"

    maker = make_failover(mk, lambda t: "good")
    checker = make_failover(lambda v: CheckerVerdict(v == "good", "policy"),
                            lambda v: CheckerVerdict(v == "good", "policy"))
    sup = DualAgentSupervisor(maker, checker, max_retries=3)
    res = sup.run("task")
    assert res["ok"] is True and res["attempts"] == 2


def test_max_retries_exhausted():
    maker = make_failover(lambda t: "always bad", lambda t: "always bad")
    checker = make_failover(lambda v: CheckerVerdict(False, "reject"),
                            lambda v: CheckerVerdict(False, "reject"))
    sup = DualAgentSupervisor(maker, checker, max_retries=2)
    res = sup.run("task")
    assert res["ok"] is False and res["attempts"] == 2


def test_failover_on_primary_outage():
    def bad(t):
        raise ProviderOutage("down")

    maker = make_failover(bad, lambda t: "fallback answer")
    checker = make_failover(lambda v: CheckerVerdict(True, "ok"), lambda v: CheckerVerdict(True, "ok"))
    sup = DualAgentSupervisor(maker, checker, max_retries=3)
    res = sup.run("task")
    assert res["ok"] is True
    assert res["maker_provider"] == "fallback"
    assert res["output"] == "fallback answer"


def test_trace_records_all_attempts():
    calls = {"n": 0}

    def mk(t):
        calls["n"] += 1
        return "x" if calls["n"] < 3 else "good"

    maker = make_failover(mk, lambda t: "good")
    checker = make_failover(lambda v: CheckerVerdict(v == "good", ""),
                            lambda v: CheckerVerdict(v == "good", ""))
    sup = DualAgentSupervisor(maker, checker, max_retries=3)
    sup.run("t")
    assert len(sup.trace) == 3
    assert sup.trace[-1]["approved"] is True
