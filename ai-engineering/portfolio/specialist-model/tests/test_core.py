import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specialist_model.core import (  # noqa: E402
    Example,
    LoraConfig,
    WandbLogger,
    build_sft_dataset,
    compare,
    run_benchmark,
    validate_lora_config,
)


def test_build_sft_dataset_formats_and_drops():
    ex = [
        Example("Summarize this", "doc", "short summary"),
        Example("", "x", ""),
        Example("   ", "y", "ok"),
    ]
    ds = build_sft_dataset(ex, min_len=1)
    assert len(ds) == 1
    assert ds[0]["messages"][0]["role"] == "user"
    assert ds[0]["messages"][1]["content"] == "short summary"


def test_lora_config_defaults_and_qlora():
    c = LoraConfig()
    assert c.is_qlora is False
    q = LoraConfig(quantization_bit=4)
    assert q.is_qlora is True
    d = c.to_dict()
    assert d["r"] == 8 and d["lora_alpha"] == 16


def test_validate_lora_config_catches_problems():
    assert validate_lora_config(LoraConfig()) == []
    bad = validate_lora_config(LoraConfig(r=0, lora_alpha=2, quantization_bit=3))
    assert any("r must be" in p for p in bad)
    assert any("quantization_bit" in p for p in bad)


def test_run_benchmark_and_compare_improvement():
    cases = [
        {"prompt": "q1", "completion": "a1", "metric": "m1"},
        {"prompt": "q2", "completion": "a2", "metric": "m2"},
    ]
    base = run_benchmark(lambda p, c: 0.5, cases, "base")
    tuned = run_benchmark(lambda p, c: 0.9, cases, "tuned")
    cmp = compare(base, tuned)
    assert cmp["improved"] is True and cmp["verdict"] == "SPECIALIST_WINS"
    assert cmp["delta"] > 0


def test_compare_regression():
    cases = [{"prompt": "q", "completion": "a", "metric": "m"}]
    base = run_benchmark(lambda p, c: 0.9, cases, "base")
    tuned = run_benchmark(lambda p, c: 0.4, cases, "tuned")
    assert compare(base, tuned)["verdict"] == "REGRESSION"


def test_wandb_logger_summary():
    lg = WandbLogger("run1")
    lg.log(0, loss=1.0)
    lg.log(1, loss=0.5)
    assert lg.summary() == {"loss": 0.5}
    assert len(lg.history) == 2
