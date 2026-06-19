"""Early / incipient detection — lead-time before functional failure (build guide §6.8).

We don't have a continuous-ramp WEC-Sim run, so we build a PROXY degradation timeline:
stitch one sea state's runs in worsening order (healthy -> PTO sev 0.8 -> 0.6 -> 0.4 -> 0.2)
into one signal of a machine degrading over time. A classifier trained on the OTHER sea
states (so this timeline is unseen) is slid along it; we record fault-probability vs time and
measure LEAD TIME = t(functional failure) - t(first confident detection).

NOTE: this is a staged proxy, not a continuous ramp. The authentic version needs a WEC-Sim
run with time-varying PTO damping (PTO-Sim) — documented as the next data step.

Usage:  py -3.13 -m tools.run_leadtime
"""
from __future__ import annotations
import glob
import os
import re

import numpy as np
import torch

from src.datasets import WindowDS
from src.ingest import load_run
from src.models import build_model
from src.train import train
from src.utils import DATA_RAW, OUTPUTS, contiguous_labels, get_logger, set_seed
from src.windowing import CHANNELS, build_dataset

log = get_logger("leadtime")

TARGET_SS = "Hs192_Tp108"          # sea state used for the degradation timeline (held out)
STAGES = ["healthy_sev100", "pto_damping_loss_sev80", "pto_damping_loss_sev60",
          "pto_damping_loss_sev40", "pto_damping_loss_sev20"]
SEV = [1.0, 0.8, 0.6, 0.4, 0.2]    # nominal severity per stage
WIN, STRIDE, TRANSIENT, DT = 384, 96, 1000, 0.1
STEADY = 2500                      # steps kept per stage after the transient


def main():
    set_seed(42)

    # --- train classifier on all sea states EXCEPT the target (timeline must be unseen) ---
    X, y_raw, runid = build_dataset(win=WIN, stride=STRIDE, transient=TRANSIENT)
    y, present = contiguous_labels(y_raw)
    n_cls = len(present)
    files = sorted(glob.glob(str(DATA_RAW / "run_*.csv")))
    ss = np.array([re.search(r"_(Hs\d+_Tp\d+)", os.path.basename(f)).group(1) for f in files])
    seastate = ss[runid]
    tr = seastate != TARGET_SS
    mu = X[tr].mean((0, 1), keepdims=True); sd = X[tr].std((0, 1), keepdims=True) + 1e-8
    n = tr.sum(); idx = np.random.permutation(n); cut = int(0.85 * n)
    Xtr = (X[tr] - mu) / sd
    model = train(build_model("cnn", len(CHANNELS), n_cls),
                  WindowDS(Xtr[idx[:cut]], y[tr][idx[:cut]]),
                  WindowDS(Xtr[idx[cut:]], y[tr][idx[cut:]]),
                  n_cls=n_cls, y_tr=y[tr][idx[:cut]], epochs=25)
    healthy_col = present.index(0)

    # --- build the degradation timeline from the target sea state's raw CSVs ---
    segs, bounds, t0 = [], [], 0
    for stage in STAGES:
        matches = glob.glob(str(DATA_RAW / f"run_*{stage}_{TARGET_SS}.csv"))
        if not matches:
            log.warning("missing stage %s @ %s", stage, TARGET_SS); continue
        df, _ = load_run(matches[0])
        seg = df[CHANNELS].to_numpy(np.float32)[TRANSIENT:TRANSIENT + STEADY]
        segs.append(seg); bounds.append((t0, t0 + len(seg))); t0 += len(seg)
    timeline = np.concatenate(segs)
    fail_step = bounds[-1][0]          # functional failure = onset of the worst stage (sev 0.2)

    # --- slide the classifier, record fault probability over time ---
    model.eval()
    times, pfault = [], []
    for s in range(0, len(timeline) - WIN, STRIDE):
        w = (timeline[s:s + WIN] - mu[0]) / sd[0]
        xt = torch.tensor(w, dtype=torch.float32).T.unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(model(xt), 1).numpy()[0]
        times.append((s + WIN // 2) * DT)
        pfault.append(1.0 - p[healthy_col])
    times, pfault = np.array(times), np.array(pfault)

    # --- first confident detection (>=0.7 for 3 consecutive windows) ---
    thr, persist, streak, detect_step = 0.7, 3, 0, None
    for i, pf in enumerate(pfault):
        streak = streak + 1 if pf >= thr else 0
        if streak >= persist and detect_step is None:
            detect_step = (i - persist + 1) * STRIDE + WIN // 2
    fail_time = fail_step * DT
    if detect_step is not None:
        detect_time = detect_step * DT
        lead = fail_time - detect_time
        log.info("first confident detection at t=%.1fs ; functional failure at t=%.1fs", detect_time, fail_time)
        log.info(">>> LEAD TIME = %.1f s (%.0f%% of the way before failure)", lead, 100 * lead / fail_time)
    else:
        log.info("no confident detection before end of timeline")
        detect_time, lead = None, None

    _plot(times, pfault, bounds, SEV, fail_time, detect_time)


def _plot(times, pfault, bounds, sev, fail_time, detect_time):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    out = OUTPUTS / "leadtime.png"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, pfault, lw=1.5, label="P(fault)")
    ax.axhline(0.7, ls=":", c="gray", label="detect threshold")
    for (a, b), s in zip(bounds, sev):
        ax.axvspan(a * 0.1, b * 0.1, alpha=0.06, color="red" if s < 1 else "green")
        ax.text((a + b) / 2 * 0.1, 1.02, f"sev {s}", ha="center", fontsize=8)
    ax.axvline(fail_time, c="red", ls="--", label="functional failure")
    if detect_time is not None:
        ax.axvline(detect_time, c="blue", ls="--", label="first detection")
        ax.annotate("", xy=(fail_time, 0.5), xytext=(detect_time, 0.5),
                    arrowprops=dict(arrowstyle="<->", color="black"))
        ax.text((detect_time + fail_time) / 2, 0.45, f"lead {fail_time-detect_time:.0f}s",
                ha="center", fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("fault probability"); ax.set_ylim(0, 1.1)
    ax.set_title(f"Incipient detection on degradation timeline ({TARGET_SS})")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout(); Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
