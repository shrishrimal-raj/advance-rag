import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

"""Approach 2 - CLIP-style shared embedding space + cosine retrieval (offline, NumPy).

Hand-defined image/text vectors in a shared space (as if produced by CLIP encoders).
For each image, rank the captions by cosine similarity. Matched pairs must outrank
mismatches - that's the whole contrastive idea.
"""

IMG = {
    "cat": np.array([0.90, 0.10, 0.05]),
    "car": np.array([0.10, 0.85, 0.05]),
    "dog": np.array([0.30, 0.20, 0.70]),
}
TXT = {
    "a photo of a cat": np.array([0.88, 0.12, 0.04]),
    "a photo of a car": np.array([0.09, 0.83, 0.06]),
    "a photo of a dog": np.array([0.28, 0.22, 0.68]),
    "a photo of a bird": np.array([0.10, 0.10, 0.10]),
}
TRUE = {"cat": "a photo of a cat", "car": "a photo of a car", "dog": "a photo of a dog"}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(img_name):
    v = IMG[img_name]
    return sorted(((cos(v, t), name) for name, t in TXT.items()), key=lambda x: -x[0])


def main():
    print("=== Approach 2: CLIP-style fusion + cosine retrieval (offline) ===")
    for img in IMG:
        ranked = retrieve(img)
        top = ranked[0][1]
        print(f"{img:4}: " + ", ".join(f"{n}({s:.3f})" for s, n in ranked[:3]))
        assert top == TRUE[img], f"{img}: expected {TRUE[img]!r}, got {top!r}"
    assert cos(IMG["cat"], TXT["a photo of a bird"]) < cos(IMG["cat"], TXT["a photo of a cat"]), "mismatch must score lower than match"
    print("self-check OK: matched text-image pairs outrank mismatches in the shared space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
