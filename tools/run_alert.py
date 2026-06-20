"""Operator early-warning ALERT system — the prognostic capstone.

Ties the pieces together on a degradation timeline (a device whose PTO worsens over time):
  1. CNN classifier      -> WHICH fault + fault probability
  2. CNN severity regressor -> HOW degraded (health indicator)
  3. trend extrapolation -> projected TIME-TO-FAILURE (severity -> failure threshold)
  4. alert logic         -> raises an operator alert with lead time + recommended action

Both models are trained on OTHER sea states, so the timeline is unseen. The degradation
timeline is a staged proxy (stitched severities) — an authentic continuous PTO-Sim ramp would
make it fully real; documented as the next data step.

Usage:  py -3.13 -m tools.run_alert
"""
from __future__ import annotations
import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ingest import load_run
from src.models import CNN1D, build_model
from src.train import train
from src.datasets import WindowDS
from src.utils import DATA_RAW, OUTPUTS, contiguous_labels, get_logger, label_names, set_seed
from src.windowing import CHANNELS, build_dataset

log = get_logger("alert")

TARGET_SS = "Hs300_Tp105"
STAGES = ["healthy_sev100", "pto_damping_loss_sev80", "pto_damping_loss_sev60",
          "pto_damping_loss_sev40", "pto_damping_loss_sev20"]
WIN, STRIDE, TRANSIENT, DT, STEADY = 384, 96, 1000, 0.1, 2500
FAIL_SEV = 0.2          # functional-failure severity threshold
P_THRESH, PERSIST = 0.7, 3


def main():
    set_seed(42)
    X, y_raw, runid = build_dataset(win=WIN, stride=STRIDE, transient=TRANSIENT)
    y, present = contiguous_labels(y_raw)
    n_cls = len(present); healthy_col = present.index(0)
    names = label_names()
    files = sorted(glob.glob(str(DATA_RAW / "run_*.csv")))
    ss = np.array([re.search(r"_(Hs\d+_Tp\d+)", os.path.basename(f)).group(1) for f in files])
    from src.ingest import parse_meta
    sev_run = np.array([parse_meta(f)["severity"] for f in files])
    seastate = ss[runid]; sev = sev_run[runid].astype(np.float32)
    tr = seastate != TARGET_SS
    mu, sd = X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-8

    # 1) classifier
    Xtr = (X[tr] - mu) / sd
    idx = np.random.permutation(tr.sum()); cut = int(0.85 * len(idx))
    clf = train(build_model("cnn", len(CHANNELS), n_cls),
                WindowDS(Xtr[idx[:cut]], y[tr][idx[:cut]]), WindowDS(Xtr[idx[cut:]], y[tr][idx[cut:]]),
                n_cls=n_cls, y_tr=y[tr][idx[:cut]], epochs=22)
    # 2) severity regressor
    reg = CNN1D(len(CHANNELS), 1)
    opt = torch.optim.Adam(reg.parameters(), 1e-3); mse = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32).transpose(1, 2)
    st = torch.tensor(sev[tr], dtype=torch.float32).unsqueeze(1)
    dl = DataLoader(TensorDataset(Xt, st), batch_size=64, shuffle=True)
    for _ in range(30):
        reg.train()
        for xb, tb in dl:
            opt.zero_grad(); mse(reg(xb), tb).backward(); opt.step()

    # build degradation timeline from the held-out sea state
    segs, bounds, t0 = [], [], 0
    for stg in STAGES:
        g = glob.glob(str(DATA_RAW / f"run_*{stg}_{TARGET_SS}.csv"))
        if not g:
            continue
        df, _ = load_run(g[0])
        seg = df[CHANNELS].to_numpy(np.float32)[TRANSIENT:TRANSIENT + STEADY]
        segs.append(seg); bounds.append((t0, t0 + len(seg))); t0 += len(seg)
    timeline = np.concatenate(segs)
    fail_time = bounds[-1][0] * DT

    clf.eval(); reg.eval()
    times, pf, psev, pcls = [], [], [], []
    for s in range(0, len(timeline) - WIN, STRIDE):
        w = (timeline[s:s + WIN] - mu[0]) / sd[0]
        xt = torch.tensor(w, dtype=torch.float32).T.unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(clf(xt), 1).numpy()[0]
            sv = float(reg(xt).item())
        times.append((s + WIN // 2) * DT); pf.append(1 - p[healthy_col])
        psev.append(sv); pcls.append(present[int(p.argmax())])
    times, pf, psev = np.array(times), np.array(pf), np.array(psev)

    # ---- alert logic ----
    streak, alert_i = 0, None
    for i, v in enumerate(pf):
        streak = streak + 1 if v >= P_THRESH else 0
        if streak >= PERSIST and alert_i is None:
            alert_i = i - PERSIST + 1
    print("\n" + "=" * 64)
    if alert_i is None:
        print("NO ALERT raised over the timeline."); return
    at = times[alert_i]
    ftype = names.get(_mode(pcls[alert_i:alert_i + 5]), "fault")
    sev_now = psev[alert_i]
    # time-to-failure: linear trend of predicted severity after alert, extrapolate to FAIL_SEV
    j = slice(alert_i, min(alert_i + 12, len(psev)))
    slope, intercept = np.polyfit(times[j], psev[j], 1)
    ttf = (FAIL_SEV - intercept) / slope - at if slope < -1e-5 else float("nan")
    print("  [!]  WEC CONDITION ALERT  (operator console)")
    print("=" * 64)
    print(f"  time of alert        : t = {at:.0f} s")
    print(f"  fault type           : {ftype}")
    print(f"  est. severity now    : {sev_now:.2f}  (1.0 healthy -> {FAIL_SEV} failure)")
    print(f"  lead time to failure : {fail_time - at:.0f} s  ({100*(fail_time-at)/fail_time:.0f}% before functional failure)")
    if not np.isnan(ttf):
        print(f"  projected TTF (trend): ~{ttf:.0f} s from alert (severity slope {slope:.2e}/s)")
    print(f"  recommended action   : schedule {ftype.split('_')[0].upper()} inspection; monitor trend")
    print("=" * 64)
    _plot(times, pf, psev, bounds, at, fail_time)


def _mode(a):
    vals, cnt = np.unique(a, return_counts=True); return vals[cnt.argmax()]


def _plot(times, pf, psev, bounds, at, fail_time):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, pf, label="P(fault)", lw=1.4)
    ax.plot(times, psev, label="health indicator (severity)", lw=1.4, color="green")
    ax.axhline(P_THRESH, ls=":", c="gray"); ax.axhline(FAIL_SEV, ls=":", c="red")
    ax.axvline(at, c="blue", ls="--", label="ALERT")
    ax.axvline(fail_time, c="red", ls="--", label="functional failure")
    ax.set_xlabel("time (s)"); ax.set_ylabel("probability / severity"); ax.set_ylim(0, 1.1)
    ax.set_title("Operator early-warning: detection + health trend + alert"); ax.legend(fontsize=8)
    fig.tight_layout(); out = OUTPUTS / "operator_alert.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
