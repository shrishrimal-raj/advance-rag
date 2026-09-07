import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json

"""Approach 2 - production readiness mechanics (offline).

Config validation (fail fast), health/readiness probes, graceful degradation when a
dependency is down, and structured JSON logging. No network needed.
"""


class ConfigError(Exception):
    pass


def validate_config(cfg):
    problems = []
    if not cfg.get("llm_api_key"):
        problems.append("missing llm_api_key")
    if cfg.get("max_retries", 0) < 0:
        problems.append("max_retries must be >= 0")
    if cfg.get("timeout_s", 0) <= 0:
        problems.append("timeout_s must be > 0")
    if problems:
        raise ConfigError("; ".join(problems))
    return True


def health():
    return {"status": "ok"}


def ready(cfg):
    try:
        validate_config(cfg)
        return {"ready": True}
    except ConfigError as e:
        return {"ready": False, "reason": str(e)}


def ask(dependency_up, question):
    """Graceful degradation: fall back instead of crashing when the dep is down."""
    if not dependency_up:
        return {"ok": False, "fallback": True, "message": "LLM unavailable; returning cached fallback."}
    return {"ok": True, "fallback": False, "message": f"answered: {question}"}


def log_event(level, event, **fields):
    return json.dumps({"level": level, "event": event, **fields}, ensure_ascii=False)


def main():
    print("=== Approach 2: production readiness (offline) ===")
    good = {"llm_api_key": "sk-x", "max_retries": 3, "timeout_s": 30}
    assert validate_config(good) is True
    bad = {"llm_api_key": "", "max_retries": -1, "timeout_s": 0}
    try:
        validate_config(bad)
        assert False, "should have raised"
    except ConfigError as e:
        print(f"config validation caught: {e}")
    print("health:", health())
    print("ready(good):", ready(good))
    print("ready(bad):", ready(bad))
    up = ask(True, "hi")
    down = ask(False, "hi")
    print("ask(up):", up)
    print("ask(down):", down)
    line = log_event("info", "request", path="/ask", ms=12)
    print("log:", line)
    assert ready(good)["ready"] is True
    assert ready(bad)["ready"] is False
    assert up["fallback"] is False and down["fallback"] is True, "must degrade gracefully"
    assert json.loads(line)["event"] == "request"
    print("self-check OK: config validated, probes correct, graceful degradation, structured log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
