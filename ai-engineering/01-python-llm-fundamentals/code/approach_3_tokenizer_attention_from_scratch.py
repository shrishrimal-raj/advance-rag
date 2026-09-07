import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 3 - tokenizer + attention FROM SCRATCH (pure numpy, no API calls).

Demystifies what the provider does internally:
  1. A tiny byte-level BPE-style tokenizer (text -> token ids -> text round-trip).
  2. Scaled dot-product attention implemented by hand (softmax(QK^T/sqrt(d)) V).

Fully local and light - safe on an 8GB laptop.
"""
import math

import numpy as np

# ---- minimal byte-level BPE-style tokenizer (teaching demo) ----
VOCAB = {bytes([i]): i for i in range(256)}   # byte -> id (0..255)
MERGES = {(97, 98): 256}                       # one learned merge: 'a'+'b' -> id 256
ID2TOK = {i: bytes([i]).decode("latin1") for i in range(256)}
ID2TOK[256] = "ab"


def tokenize(text: str) -> list[int]:
    ids = [VOCAB[bytes([c])] for c in text.encode("latin1", errors="ignore")]
    changed = True
    while changed:
        changed = False
        new: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) in MERGES:
                new.append(MERGES[(ids[i], ids[i + 1])])
                i += 2
                changed = True
            else:
                new.append(ids[i])
                i += 1
        ids = new
    return ids


def detokenize(ids: list[int]) -> str:
    return "".join(ID2TOK.get(i, "?") for i in ids)


# ---- scaled dot-product attention from scratch ----
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / math.sqrt(d_k)
    weights = softmax(scores)
    out = weights @ V
    return out, weights


def main() -> None:
    print("=== Approach 3: tokenizer + attention from scratch (no API) ===")
    print()

    text = "ab ab cd"
    ids = tokenize(text)
    back = detokenize(ids)
    print(f"text       = {text!r}")
    print(f"token ids  = {ids}")
    print(f"detokenize = {back!r}")
    assert back == text, "tokenizer round-trip failed"
    print("round-trip OK")
    print()

    rng = np.random.default_rng(0)
    n, d = 4, 8
    Q = rng.normal(size=(1, n, d))
    K = rng.normal(size=(1, n, d))
    V = rng.normal(size=(1, n, d))
    out, w = attention(Q, K, V)
    print("attention weights (each row sums to 1):")
    print(np.round(w[0], 3))
    assert np.allclose(w[0].sum(axis=1), 1.0, atol=1e-6), "weights must sum to 1"
    assert out.shape == (1, n, d), "output shape mismatch"
    print(f"output shape = {out.shape}")
    print("attention OK")


if __name__ == "__main__":
    main()
