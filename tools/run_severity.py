"""Severity / health-indicator regression — the prognostic core.

The classifier says WHICH fault. This regressor estimates HOW degraded the device is: it
predicts the severity multiplier per window (1.0 = healthy, lower = more degraded). That gives
a continuous health indicator the operator can trend over time to forecast time-to-failure.

We report MAE and, crucially, that predicted severity tracks true severity monotonically
(so a worsening trend is real, not noise). Run-aware split.

Usage:  py -3.13 -m tools.run_severity
"""
from __future__ import annotations
import glob

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from src.ingest import parse_meta
from src.models import CNN1D
from src.utils import DATA_RAW, OUTPUTS, get_logger, set_seed
from src.windowing import CHANNELS, build_dataset

log = get_logger("severity")


def main():
    set_seed(42)
    X, y, runid = build_dataset(win=384, stride=96, transient=1000)
    files = sorted(glob.glob(str(DATA_RAW / "run_*.csv")))
    sev_run = np.array([parse_meta(f)["severity"] for f in files])
    sev = sev_run[runid].astype(np.float32)            # per-window severity target

    gss = GroupShuffleSplit(1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups=runid))
    mu, sd = X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-8
    Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32).transpose(1, 2)
    Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32).transpose(1, 2)
    str_, ste = torch.tensor(sev[tr]).unsqueeze(1), torch.tensor(sev[te]).unsqueeze(1)

    model = CNN1D(len(CHANNELS), 1)                     # 1 output = regression
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    crit = nn.MSELoss()
    dl = DataLoader(TensorDataset(Xtr, str_), batch_size=64, shuffle=True)
    for ep in range(30):
        model.train()
        for xb, tb in dl:
            opt.zero_grad(); loss = crit(model(xb), tb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte).squeeze(1).numpy()
    true = sev[te]

    mae = float(np.abs(pred - true).mean())
    log.info("severity regression MAE = %.3f (severity in [0.2, 1.0])", mae)
    print("\n=== predicted severity vs true (should be monotonic) ===")
    print(f"{'true sev':>10}{'pred mean':>11}{'pred std':>10}{'n':>7}")
    for s in sorted(set(np.round(true, 2))):
        m = np.abs(true - s) < 1e-6
        print(f"{s:>10.2f}{pred[m].mean():>11.3f}{pred[m].std():>10.3f}{m.sum():>7}")

    # rank correlation (monotonic trend quality)
    from scipy.stats import spearmanr
    rho = spearmanr(true, pred).statistic
    print(f"\nSpearman rank corr (trend fidelity) = {rho:.3f}  (1.0 = perfectly monotonic)")
    _plot(true, pred)


def _plot(true, pred):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true, pred, s=6, alpha=0.3)
    ax.plot([0.1, 1.05], [0.1, 1.05], "r--", label="ideal")
    ax.set_xlabel("true severity (1=healthy)"); ax.set_ylabel("predicted severity")
    ax.set_title("Health indicator: predicted vs true degradation"); ax.legend()
    fig.tight_layout(); out = OUTPUTS / "severity_regression.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
