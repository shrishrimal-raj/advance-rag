import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Audited Tool Bridge. Import-safe (builds only if fastapi present)."""
from audited_tool_bridge.core import ToolBridge

try:
    from fastapi import FastAPI, Header
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False

BRIDGE = ToolBridge(valid_keys={"demo-key": "demo-client"}, rate_capacity=10, rate_refill_per_sec=2.0)
BRIDGE.register_tool("echo", lambda text: text)
BRIDGE.register_tool("sum", lambda a, b: a + b)


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Audited Tool Bridge")

    class ToolReq(BaseModel):
        args: dict = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "tools": list(BRIDGE.tools)}

    @app.post("/tools/{name}")
    def call_tool(name: str, req: ToolReq, x_api_key: str = Header(default="")):
        return BRIDGE.invoke(x_api_key, name, req.args)

    @app.get("/audit")
    def audit(client: str = None):
        return BRIDGE.audit_log(client)

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn audited_tool_bridge.app:app --port 8000")
