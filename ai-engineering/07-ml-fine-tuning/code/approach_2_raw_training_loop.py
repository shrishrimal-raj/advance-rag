import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

"""Approach 2 - the raw training loop in PyTorch (offline, tiny).

Trains a 2-layer MLP on a synthetic task to make forward/loss/backward/update
concrete. Logs train+val loss per epoch and asserts convergence. No downloads.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


def make_data(n=400):
    x = torch.randn(n, 4)
    y = torch.sigmoid(x[:, 0] + 0.5 * x[:, 1] ** 2)  # deterministic target in [0,1]
    idx = torch.randperm(n)
    cut = int(0.8 * n)
    return x[idx[:cut]], y[idx[:cut]], x[idx[cut:]], y[idx[cut:]]


def main():
    print("=== Approach 2: raw PyTorch training loop (offline) ===")
    Xtr, ytr, Xva, yva = make_data()
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = nn.MSELoss()
    hist = []
    for epoch in range(60):
        model.train()
        loss = lossf(model(Xtr).squeeze(-1), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(Xva).squeeze(-1), yva))
        hist.append((float(loss.detach()), vloss))
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d}  train={hist[-1][0]:.4f}  val={vloss:.4f}")
    first_val = hist[0][1]
    last_val = hist[-1][1]
    reduction = (first_val - last_val) / first_val if first_val else 0.0
    print(f"\nval loss: {first_val:.4f} -> {last_val:.4f}  (reduction {reduction*100:.0f}%)")
    assert last_val < first_val, "model must converge (val loss must drop)"
    assert reduction > 0.3, f"toy task should improve substantially, got {reduction*100:.0f}%"
    print("self-check OK: forward/loss/backward/update converges on the toy task")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
