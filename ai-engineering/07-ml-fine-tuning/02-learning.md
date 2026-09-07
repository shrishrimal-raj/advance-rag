# 📖 Week 7 Learning — ML & Fine-Tuning

> Read top-to-bottom. Each section ends with a **"why it matters"** line. **Diagrams: Noob → Expert.**
> ⚠️ This week is **code + docs** on this machine: real fine-tuning is too heavy for an 8 GB laptop. The runnable parts prove the *mechanics* on a toy problem.

---

## 1. Fine-Tuning vs RAG

Two ways to make a model domain-specific:
- **RAG** — keep the model frozen; inject knowledge at inference. Great for *changing* knowledge, citations, and low cost.
- **Fine-tuning** — update the model's weights on your data. Great for *behavior/format/style* and removing retrieval latency.

Rule of thumb: **RAG for knowledge, fine-tuning for behavior.** If the answer is "the model keeps getting the format wrong," fine-tune. If it's "the model doesn't know our latest facts," use RAG.

**Why it matters:** picking the wrong tool wastes weeks. Most teams should try RAG first.

---

## 2. Full vs Parameter-Efficient (PEFT/LoRA)

Full fine-tuning updates *every* weight - expensive and prone to overfitting on small data. **PEFT** freezes the base model and trains only a small set of added parameters. **LoRA** injects low-rank matrices into attention layers; you train ~0.1–1% of parameters. Cheaper, faster, less overfitting, and you can swap adapters per task.

**Why it matters:** PEFT is why fine-tuning is affordable on modest hardware and why one base model can serve many tasks.

---

## 3. Dataset Preparation

Quality beats quantity. You need clean **prompt -> completion** pairs in the exact template your trainer expects, a **train/val split**, and enough examples to avoid memorization. Common failure: leaking val rows into train, or a template mismatch that silently trains garbage.

**Why it matters:** most "bad fine-tune" results are dataset bugs, not algorithm bugs.

---

## 4. The Training Loop

Every epoch: **forward** pass (predict) -> **loss** (how wrong) -> **backward** (gradients) -> **optimizer.step** (update weights). Track train + val loss; if val loss rises while train falls, you're **overfitting** - stop early. This loop is identical whether you're training a 2-layer MLP or a 7B LLM.

**Why it matters:** understanding forward/loss/backward/update demystifies every framework's trainer.

---

## 5. Evaluating a Fine-Tuned Model

Don't trust train loss alone. Evaluate on held-out data with task metrics (accuracy, BLEU/ROUGE for text, or your RAGAS-style checks). Compare against the base model and against RAG on the same set before deciding to ship.

**Why it matters:** a fine-tune that doesn't beat your RAG baseline isn't worth its serving cost.

---

## 🧠 Diagrams: Noob → Expert

### Level 1 — Noob: one knob
```mermaid
flowchart LR
    D["data"] --> T["train"] --> M["better model"]
```

### Level 2 — Practitioner: the training loop
```mermaid
flowchart TB
    F["forward: predict"] --> L["loss"]
    L --> B["backward: gradients"]
    B --> U["optimizer.step: update"]
    U --> C{"epoch done?"}
    C -- no --> F
    C -- yes --> EV["eval on val set"]
```

### Level 3 — Expert: full vs PEFT
```mermaid
flowchart TB
    subgraph FULL["full fine-tune"]
        W["all weights trainable"]
    end
    subgraph PEFT["PEFT / LoRA"]
        FZ["base frozen"] --> AD["low-rank adapters trainable"]
    end
```

### Level 4 — Overfitting signature
```mermaid
xychart-beta
    title "train vs val loss"
    x-axis [1,2,3,4,5]
    y-axis "loss" 0 --> 1
    line [0.9,0.5,0.3,0.2,0.1]
    line [0.9,0.6,0.45,0.5,0.6]
```

### Level 5 — RAG vs fine-tune decision
```mermaid
flowchart TD
    Q["need domain-specific?"] --> K{"changing knowledge?"}
    K -- yes --> RAG["RAG"]
    K -- no --> B{"format/behavior?"}
    B -- yes --> FT["fine-tune (PEFT)"]
    B -- no --> KEEP["leave as-is"]
```

### Level 6 — End-to-end fine-tune pipeline
```mermaid
flowchart LR
    RAW["raw examples"] --> PREP["dataset prep + split"]
    PREP --> CFG["LoRA config"]
    CFG --> LOOP["train loop"]
    LOOP --> EVAL["eval vs base + RAG"]
    EVAL --> SHIP{"beats baseline?"}
    SHIP -- yes --> DEPLOY["ship adapter"]
```

---

## Key Terms (ubiquitous language)
| Term | Meaning |
|------|---------|
| Fine-tuning | Updating weights on domain data |
| PEFT | Train few params, freeze the rest |
| LoRA | Low-rank adapter matrices |
| Epoch | One full pass over the training set |
| Loss | How wrong the predictions are |
| Backprop | Computing gradients for the update |
| Overfitting | Train improves, validation worsens |
| Adapter | Swappable per-task PEFT module |
