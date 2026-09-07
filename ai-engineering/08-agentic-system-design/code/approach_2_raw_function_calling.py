import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 2 - the function-calling contract, hand-rolled (offline).

Defines tools with JSON schemas, a stand-in 'model' that emits a tool call, and a
harness that parses, validates, dispatches to the real function, and feeds the
result back. Shows the exact shape of a function call and its round-trip.
"""

TOOLS = {
    "add": {
        "description": "Add two numbers.",
        "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
        "fn": lambda a, b: a + b,
    },
    "lookup": {
        "description": "Look up a term.",
        "inputSchema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]},
        "fn": lambda term: {"refund": "30 days", "warranty": "12 months"}.get(term.lower(), "unknown"),
    },
}


def fake_model(history):
    calls = [h for h in history if h["role"] == "tool"]
    if not calls:
        return {"role": "assistant", "tool_call": {"name": "add", "arguments": {"a": 2, "b": 3}}}
    return {"role": "assistant", "content": f"The result is {calls[-1]['output']}."}


def validate(name, args):
    schema = TOOLS[name]["inputSchema"]
    for req in schema.get("required", []):
        if req not in args:
            return f"missing required arg: {req}"
    return None


def harness(model, max_turns=3):
    history = [{"role": "user", "content": "Add 2 and 3."}]
    for _ in range(max_turns):
        msg = model(history)
        if "tool_call" in msg:
            tc = msg["tool_call"]
            err = validate(tc["name"], tc["arguments"])
            if err:
                history.append({"role": "tool", "name": tc["name"], "output": f"ERROR: {err}"})
                continue
            out = TOOLS[tc["name"]]["fn"](**tc["arguments"])
            history.append({"role": "tool", "name": tc["name"], "output": out})
        else:
            return msg["content"], history
    return "(max turns)", history


def main():
    print("=== Approach 2: function-calling contract (offline) ===")
    answer, history = harness(fake_model)
    for h in history:
        if h["role"] == "tool":
            print(f"  tool {h['name']} -> {h['output']}")
    print(f"final: {answer}")
    assert answer == "The result is 5.", f"expected sum, got {answer!r}"
    bad = validate("add", {"a": 1})
    assert bad == "missing required arg: b", "must reject missing required arg"
    print("self-check OK: round-trip works and invalid args are rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
