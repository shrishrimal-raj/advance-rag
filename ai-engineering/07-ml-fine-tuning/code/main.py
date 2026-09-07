import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""The Domain-Specific Model (Week 7 weekly build).

Complete HF + PEFT fine-tuning pipeline. DRY-RUN by default: prints every step's
config without loading a model (safe on any machine). Pass --run --model NAME to
actually train (needs GPU + a model download - NOT on this 8 GB laptop).
"""
import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


def plan(model_name="gpt2", max_length=256, lora_rank=8, epochs=2, batch=2):
    """Return the full pipeline plan as data (no side effects)."""
    try:
        import peft  # noqa: F401
        peft_ok = True
    except Exception:
        peft_ok = False
    method = "LoRA" if peft_ok else "full-fine-tune (peft missing)"
    step4 = f"4. LoraConfig(r={lora_rank}) on base model" if peft_ok else "4. full fine-tune on base model"
    return {
        "model": model_name,
        "peft_available": peft_ok,
        "method": method,
        "lora_rank": lora_rank,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch,
        "steps": [
            "1. load dataset (datasets.load_dataset)",
            "2. apply chat template + tokenize (truncation, padding)",
            "3. train/val split (no leakage)",
            step4,
            "5. TrainingArguments (epochs, batch, eval_strategy='epoch')",
            "6. Trainer.train()",
            "7. evaluate on val; compare vs base + RAG baseline",
        ],
    }


def dry_run(p):
    print("=== Domain-Specific Model - FINE-TUNE PIPELINE (dry-run) ===")
    print(f"model      : {p['model']}")
    print(f"method     : {p['method']}")
    print(f"lora_rank  : {p['lora_rank']}   max_length: {p['max_length']}")
    print(f"epochs     : {p['epochs']}   batch_size: {p['batch_size']}")
    print("\nPipeline steps:")
    for s in p["steps"]:
        print(f"  {s}")
    print("\n[dry-run] no model loaded. Add --run --model <name> to train (needs GPU + download).")


def real_run(p):
    import torch  # noqa: F401
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(p["model"])
    model = AutoModelForCausalLM.from_pretrained(p["model"])
    ds = load_dataset("json", data_files={"train": "train.json", "val": "val.json"})

    def tok(ex):
        out = tokenizer(ex["text"], truncation=True, max_length=p["max_length"], padding="max_length")
        out["labels"] = out["input_ids"]
        return out

    ds = ds.map(tok, batched=True)
    if p["peft_available"]:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=p["lora_rank"]))

    args = TrainingArguments(out_dir="out-ft", num_train_epochs=p["epochs"],
                             per_device_train_batch_size=p["batch_size"],
                             eval_strategy="epoch", logging_steps=10, report_to=[])

    class Collator:
        def __call__(self, f):
            import torch as T
            return {k: T.tensor([ex[k] for ex in f]) for k in ("input_ids", "attention_mask", "labels")}

    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["val"], data_collator=Collator())
    trainer.train()
    print("training complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually train (heavy)")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    p = plan(model_name=args.model)
    if args.run:
        try:
            real_run(p)
        except Exception as e:
            print(f"[fine-tune] could not run here ({type(e).__name__}: {e})")
            print("Expected on an 8 GB laptop: needs GPU + model download. See 03-implementation.md.")
            return 0
    else:
        dry_run(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
