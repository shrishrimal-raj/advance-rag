import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

"""Approach 3 - speech front-end from scratch (offline, NumPy).

Synthesize a waveform with two tone bursts separated by silence, compute an energy
envelope, and segment voiced regions by thresholding (a minimal VAD). Shows the
STT front-end without any model.
"""
SR = 8000
FRAME = 160


def make_signal():
    def sil(d):
        return np.zeros(int(SR * d))

    def tone(d, f=440):
        t = np.arange(int(SR * d)) / SR
        return 0.5 * np.sin(2 * np.pi * f * t)

    return np.concatenate([sil(0.3), tone(0.4), sil(0.3), tone(0.2), sil(0.3)])


def energy_envelope(x, frame=FRAME):
    n = len(x) // frame * frame
    frames = x[:n].reshape(-1, frame)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def segment(env, thresh=0.05, min_frames=3):
    voiced = env > thresh
    segs = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_frames:
                segs.append((start, i))
            start = None
    if start is not None and len(voiced) - start >= min_frames:
        segs.append((start, len(voiced)))
    return [(s * FRAME / SR, e * FRAME / SR) for s, e in segs]


def main():
    print("=== Approach 3: speech front-end (offline) ===")
    x = make_signal()
    env = energy_envelope(x)
    segs = segment(env)
    print(f"signal length={len(x)/SR:.2f}s  envelope frames={len(env)}")
    for i, (s, e) in enumerate(segs):
        print(f"  segment {i+1}: {s:.2f}s -> {e:.2f}s  (dur {e-s:.2f}s)")
    assert len(segs) == 2, f"expected 2 voiced segments, got {len(segs)}"
    durs = [e - s for s, e in segs]
    assert durs[0] >= 0.3 and durs[1] >= 0.1, f"durations off: {[round(d,2) for d in durs]}"
    print("self-check OK: energy envelope + VAD finds the two tone bursts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
