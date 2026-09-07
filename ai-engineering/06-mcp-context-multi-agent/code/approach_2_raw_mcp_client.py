import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 2 - the MCP protocol, hand-rolled (no SDK, fully offline).

Implements a minimal in-process JSON-RPC MCP server (one tool: lookup) and a
client that performs the real handshake: initialize -> notifications/initialized
-> tools/list -> tools/call. Shows the exact message shapes MCP uses.
"""

KB = {"refund": "30 days full refund", "warranty": "12 months"}


class McpServer:
    """Minimal in-process MCP server speaking JSON-RPC over a send() boundary."""

    def __init__(self):
        self.tools = {
            "lookup": {
                "name": "lookup",
                "description": "Look up a term in a tiny knowledge base.",
                "inputSchema": {"type": "object", "properties": {"term": {"type": "string"}}, "required": ["term"]},
            }
        }

    def _handle(self, msg):
        method = msg.get("method")
        params = msg.get("params", {})
        if method == "initialize":
            return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "mini-mcp", "version": "0.1"}}
        if method == "notifications/initialized":
            return None  # notification: no response
        if method == "tools/list":
            return {"tools": list(self.tools.values())}
        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            if name != "lookup":
                raise ValueError(f"unknown tool {name}")
            term = args.get("term", "")
            value = KB.get(term.lower(), f"no entry for {term!r}")
            return {"content": [{"type": "text", "text": value}]}
        raise ValueError(f"unknown method {method}")

    def send(self, msg):
        is_notification = "id" not in msg
        result = self._handle(msg)
        if is_notification or result is None:
            return None
        return {"jsonrpc": "2.0", "id": msg["id"], "result": result}


def client_session():
    server = McpServer()
    trace = []
    counter = {"id": 1}

    def rpc(method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"] = counter["id"]
            counter["id"] += 1
        resp = server.send(msg)
        trace.append((msg, resp))
        return resp

    init = rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "mini-client", "version": "0.1"}})
    rpc("notifications/initialized", notify=True)
    tools = rpc("tools/list")["result"]["tools"]
    call = rpc("tools/call", {"name": "lookup", "arguments": {"term": "refund"}})["result"]
    return init, tools, call, trace


def main():
    print("=== Approach 2: hand-rolled MCP JSON-RPC (offline) ===")
    init, tools, call, trace = client_session()
    print(f"initialize -> protocolVersion={init['result']['protocolVersion']} capabilities={list(init['result']['capabilities'])}")
    print(f"tools/list -> {[t['name'] for t in tools]}")
    print(f"tools/call lookup(refund) -> {call['content'][0]['text']}")
    assert init["result"]["protocolVersion"], "initialize must return a protocol version"
    assert [t["name"] for t in tools] == ["lookup"], "tools/list must expose lookup"
    assert call["content"][0]["text"] == KB["refund"], "tools/call must return the KB value"
    notif = [m for m, r in trace if m["method"] == "notifications/initialized"][0]
    assert "id" not in notif, "notifications carry no id"
    print("self-check OK: full MCP handshake + tool call round-trips correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
