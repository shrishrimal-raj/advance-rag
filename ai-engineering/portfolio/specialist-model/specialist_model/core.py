"""The Specialist Model core.

The **control plane** around a LoRA/QLoRA fine-tune: dataset engineering, LoRA/QLoRA
config generation + validation, a base-vs-tuned benchmark harness, and W&B-style metric
logging. The heavy PEFT/QLoRA training itself is NOT run here (8 GB laptop constraint);
this core is the fully-offline, testable layer that drives and verifies it.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Example:
    instruction: str
    input: str = ""
    output: str = ""


def build_sft_dataset(examples: List[Example], min_len: int = 1) -> List[dict]:
    """Format examples into chat-style SFT records; drop empties below min_len."""
    out = []
    for e in examples:
        if len(e.instruction.strip()) < min_len or not e.output.strip():
            continue
        user = (e.input + "\n" if e.input else "") + e.instruction
        out.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": e.output},
        ]})
    return out


@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    quantization_bit: Optional[int] = None  # None=full LoRA, 4/8=QLoRA
    task_type: str = "CAUSAL_LM"

    @property
    def is_qlora(self) -> bool:
        return self.quantization_bit in (4, 8)

    def to_dict(self) -> dict:
        return {"r": self.r, "lora_alpha": self.lora_alpha, "lora_dropout": self.lora_dropout,
                "target_modules": list(self.target_modules), "quantization_bit": self.quantization_bit,
                "task_type": self.task_type, "is_qlora": self.is_qlora}


def validate_lora_config(cfg: LoraConfig) -> List[str]:
    """Return a list of problems (empty = valid)."""
    problems = []
    if cfg.r <= 0:
        problems.append("r must be > 0")
    if cfg.lora_alpha < cfg.r:
        problems.append("lora_alpha should be >= r")
    if not (0.0 <= cfg.lora_dropout < 1.0):
        problems.append("lora_dropout must be in [0,1)")
    if not cfg.target_modules:
        problems.append("target_modules must be non-empty")
    if cfg.quantization_bit is not None and cfg.quantization_bit not in (4, 8):
        problems.append("quantization_bit must be 4, 8, or None")
    return problems


@dataclass
class BenchmarkResult:
    model: str
    scores: Dict[str, float]
    mean: float


def run_benchmark(scorer: Callable[[str, str], float], cases: List[dict], model_name: str) -> BenchmarkResult:
    """Score each case with `scorer(prompt, completion)` in [0,1]; return per-metric + mean."""
    scores = {}
    for c in cases:
        scores[c["metric"]] = scorer(c["prompt"], c["completion"])
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    return BenchmarkResult(model=model_name, scores=scores, mean=mean)


def compare(base: BenchmarkResult, tuned: BenchmarkResult) -> dict:
    """Head-to-head: does the specialist beat the base model?"""
    delta = round(tuned.mean - base.mean, 6)
    return {
        "base_mean": round(base.mean, 6),
        "tuned_mean": round(tuned.mean, 6),
        "delta": delta,
        "improved": delta > 0,
        "verdict": "SPECIALIST_WINS" if delta > 0 else ("TIE" if delta == 0 else "REGRESSION"),
    }


class WandbLogger:
    """W&B-style metric logger (in-memory stand-in; swap in real wandb in prod)."""

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.history: List[dict] = []

    def log(self, step: int, **metrics):
        self.history.append({"step": step, **metrics})

    def summary(self) -> dict:
        if not self.history:
            return {}
        last = self.history[-1]
        return {k: v for k, v in last.items() if k != "step"}
