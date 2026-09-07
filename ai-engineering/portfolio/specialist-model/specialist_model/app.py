import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""FastAPI wrapper for The Specialist Model. Import-safe (builds only if fastapi present).

This is the control plane: dataset formatting, LoRA config validation, and benchmarking.
The heavy QLoRA training runs on cloud GPUs, never on the dev laptop.
"""
from specialist_model.core import (
    Example,
    LoraConfig,
    build_sft_dataset,
    compare,
    run_benchmark,
    validate_lora_config,
)

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _HAVE_FASTAPI = True
except Exception:  # pragma: no cover
    _HAVE_FASTAPI = False


def _overlap(p: str, c: str) -> float:
    pt = set(p.lower().split())
    ct = set(c.lower().split())
    return len(pt & ct) / len(pt) if pt else 0.0


def build_app():
    if not _HAVE_FASTAPI:
        return None
    app = FastAPI(title="The Specialist Model")

    class ExIn(BaseModel):
        instruction: str
        input: str = ""
        output: str = ""

    class DatasetReq(BaseModel):
        examples: list
        min_len: int = 1

    class LoraReq(BaseModel):
        r: int = 8
        lora_alpha: int = 16
        lora_dropout: float = 0.05
        target_modules: list = None
        quantization_bit: int = None

    class CaseIn(BaseModel):
        prompt: str
        completion: str
        metric: str

    class BenchReq(BaseModel):
        base_cases: list
        tuned_cases: list

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "note": "control plane; heavy training runs on cloud GPUs"}

    @app.post("/dataset")
    def dataset(req: DatasetReq):
        ex = [Example(e.instruction, e.input, e.output) for e in req.examples]
        return {"count": len(build_sft_dataset(ex, req.min_len))}

    @app.post("/config")
    def config(req: LoraReq):
        cfg = LoraConfig(r=req.r, lora_alpha=req.lora_alpha, lora_dropout=req.lora_dropout,
                         target_modules=req.target_modules or ["q_proj", "v_proj"],
                         quantization_bit=req.quantization_bit)
        return {"config": cfg.to_dict(), "problems": validate_lora_config(cfg)}

    @app.post("/benchmark")
    def bench(req: BenchReq):
        base = run_benchmark(_overlap, [{"prompt": c.prompt, "completion": c.completion, "metric": c.metric} for c in req.base_cases], "base")
        tuned = run_benchmark(_overlap, [{"prompt": c.prompt, "completion": c.completion, "metric": c.metric} for c in req.tuned_cases], "tuned")
        return compare(base, tuned)

    return app


app = build_app()

if __name__ == "__main__":
    print("fastapi not installed" if app is None else "Run: uvicorn specialist_model.app:app --port 8000")
