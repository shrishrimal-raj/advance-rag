import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_gateway.gateway import AIGateway, GatewayConfig, ProviderError  # noqa: E402


def ok_provider(p):
    return "ok:" + p


def bad_provider(p):
    raise ProviderError("always down")


def flaky_then_ok(p):
    flaky_then_ok.calls = getattr(flaky_then_ok, "calls", 0) + 1
    if flaky_then_ok.calls == 1:
        raise ProviderError("boom")
    return "recovered:" + p


def test_first_provider_success():
    g = AIGateway([ok_provider])
    r = g.complete("hi")
    assert r.ok and r.provider == "ok_provider" and r.text == "ok:hi"


def test_fallback_to_second():
    g = AIGateway([bad_provider, ok_provider])
    r = g.complete("hi")
    assert r.ok and r.provider == "ok_provider"


def test_retry_recovers():
    flaky_then_ok.calls = 0
    g = AIGateway([flaky_then_ok], GatewayConfig(max_retries=2))
    r = g.complete("hi")
    assert r.ok and r.text == "recovered:hi" and r.attempts == 2


def test_all_fail_degrades():
    g = AIGateway([bad_provider])
    r = g.complete("hi")
    assert not r.ok and r.degraded


def test_rate_limit():
    g = AIGateway([ok_provider], GatewayConfig(rate_limit_per_min=1))
    assert g.complete("a").ok
    limited = g.complete("b")
    assert not limited.ok and limited.degraded


def test_config_validation():
    with pytest.raises(ValueError):
        AIGateway([ok_provider], GatewayConfig(max_retries=-1))
