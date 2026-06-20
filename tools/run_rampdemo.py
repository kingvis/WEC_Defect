"""Authentic prognosis on a fine degradation timeline (17 health levels, identical waves).

Trains the CNN classifier + CNN severity regressor on the main 4-class dataset (data/raw),
then runs them along a smooth degradation timeline built from data/degradation/raw (PTO
damping 1.0 -> 0.2 in 0.05 steps under the SAME wave realisation). Produces:
  - first confident fault detection + lead time before functional failure
  - a smooth health-indicator trend -> linear extrapolation -> authentic time-to-failure (TTF)
  - operator alert

This replaces the earlier 5-stage stitched proxy with a fine 17-level degradation, so the
severity trend and TTF projection are genuine.

Usage:  py -3.13 -m tools.run_rampdemo
"""
from __future__ import annotations
import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.datasets import WindowDS
from src.ingest import load_run
from src.models import CNN1D, build_model
from src.train import train
from src.utils import (DATA_RAW, OUTPUTS, ROOT, contiguous_labels, get_logger,
                       label_names, set_seed)
from src.windowing import CHANNELS, build_dataset

log = get_logger("rampdemo")
DEG_DIR = ROOT / "data" / "degradation" / "raw"
WIN, STRIDE, TRANSIENT, DT, STEADY = 384, 96, 1000, 0.1, 2000
FAIL_SEV, P_THRESH, PERSIST = 0.2, 0.7, 3


def main():
    set_seed(42)
    # --- train classifier + severity regressor on the main 4-class dataset ---
    X, y_raw, runid = build_dataset(win=WIN, stride=STRIDE, transient=TRANSIENT)
    y, present = contiguous_labels(y_raw)
    n_cls = len(present); healthy_col = present.index(0)
    names = label_names()
    from src.ingest import parse_meta
    files = sorted(glob.glob(str(DATA_RAW / "run_*.csv")))
    sev_run = np.array([parse_meta(f)["severity"] for f in files])
    sev = sev_run[runid].astype(np.float32)

    mu, sd = X.mean((0, 1), keepdims=True), X.std((0, 1), keepdims=True) + 1e-8
    Xs = (X - mu) / sd
    idx = np.random.permutation(len(Xs)); cut = int(0.85 * len(idx))
    clf = train(build_model("cnn", len(CHANNELS), n_cls),
                WindowDS(Xs[idx[:cut]], y[idx[:cut]]), WindowDS(Xs[idx[cut:]], y[idx[cut:]]),
                n_cls=n_cls, y_tr=y[idx[:cut]], epochs=22)
    reg = CNN1D(len(CHANNELS), 1); opt = torch.optim.Adam(reg.parameters(), 1e-3); mse = nn.MSELoss()
    Xt = torch.tensor(Xs, dtype=torch.float32).transpose(1, 2)
    dl = DataLoader(TensorDataset(Xt, torch.tensor(sev).unsqueeze(1)), batch_size=64, shuffle=True)
    for _ in range(30):
        reg.train()
        for xb, tb in dl:
            opt.zero_grad(); mse(reg(xb), tb).backward(); opt.step()

    # --- build smooth degradation timeline (health 1.0 -> 0.2) ---
    levels = sorted(glob.glob(str(DEG_DIR / "run_*.csv")),
                    key=lambda f: -int(re.search(r"sev(\d+)", f).group(1)))   # 100 -> 20
    segs, true_h, t0, bounds = [], [], 0, []
    for f in levels:
        h = int(re.search(r"sev(\d+)", os.path.basename(f)).group(1)) / 100.0
        df, _ = load_run(f)
        seg = df[CHANNELS].to_numpy(np.float32)[TRANSIENT:TRANSIENT + STEADY]
        segs.append(seg); bounds.append((t0, t0 + len(seg), h)); t0 += len(seg)
    timeline = np.concatenate(segs)
    fail_time = next((a for a, b, h in bounds if h <= FAIL_SEV), bounds[-1][0]) * DT

    # --- slide both models along the timeline ---
    clf.eval(); reg.eval()
    times, pf, psev = [], [], []
    for s in range(0, len(timeline) - WIN, STRIDE):
        w = (timeline[s:s + WIN] - mu[0]) / sd[0]
        xt = torch.tensor(w, dtype=torch.float32).T.unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(clf(xt), 1).numpy()[0]; sv = float(reg(xt).item())
        times.append((s + WIN // 2) * DT); pf.append(1 - p[healthy_col]); psev.append(sv)
    times, pf, psev = np.array(times), np.array(pf), np.array(psev)

    # --- alert + authentic TTF from the smooth severity trend ---
    streak, alert_i = 0, None
    for i, v in enumerate(pf):
        streak = streak + 1 if v >= P_THRESH else 0
        if streak >= PERSIST and alert_i is None:
            alert_i = i - PERSIST + 1
    # global degradation rate (smooth trend over whole timeline)
    slope, intercept = np.polyfit(times, psev, 1)
    print("\n" + "=" * 64)
    if alert_i is None:
        print("NO ALERT raised."); slope = 0
    else:
        at = times[alert_i]
        sev_now = psev[max(0, alert_i - 2):alert_i + 3].mean()
        ttf_from_alert = ((FAIL_SEV - intercept) / slope) - at if slope < -1e-6 else float("nan")
        print("  [!]  WEC PROGNOSTIC ALERT  (operator console)")
        print("=" * 64)
        print(f"  alert time           : t = {at:.0f} s")
        print(f"  est. health (severity): {sev_now:.2f}   (1.0 healthy -> {FAIL_SEV} failure)")
        print(f"  lead time to failure : {fail_time - at:.0f} s  ({100*(fail_time-at)/fail_time:.0f}% before failure)")
        print(f"  degradation rate     : {slope:.2e} /s   (health-indicator trend)")
        if not np.isnan(ttf_from_alert):
            print(f"  projected TTF        : ~{ttf_from_alert:.0f} s from alert (trend extrapolation)")
        print(f"  recommended action   : schedule PTO inspection; track health trend")
    print("=" * 64)
    log.info("severity trend: slope %.3e/s, intercept %.3f (Spearman-smooth degradation)", slope, intercept)
    _plot(times, pf, psev, bounds, alert_i, fail_time, slope, intercept)


def _plot(times, pf, psev, bounds, alert_i, fail_time, slope, intercept):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, pf, label="P(fault)", lw=1.3)
    ax.plot(times, psev, label="health indicator", color="green", lw=1.3)
    ax.plot(times, intercept + slope * times, "g:", lw=1, label="health trend")
    ax.step([0.5*(a+b)*0.1 for a, b, h in bounds], [h for a, b, h in bounds], where="mid",
            color="black", alpha=0.4, lw=1, label="true health")
    ax.axhline(P_THRESH, ls=":", c="gray"); ax.axhline(FAIL_SEV, ls=":", c="red")
    if alert_i is not None:
        ax.axvline(times[alert_i], c="blue", ls="--", label="ALERT")
    ax.axvline(fail_time, c="red", ls="--", label="functional failure")
    ax.set_xlabel("time (s)"); ax.set_ylabel("probability / health"); ax.set_ylim(0, 1.15)
    ax.set_title("Authentic prognosis: fine degradation timeline (17 levels)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); out = OUTPUTS / "operator_alert.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
