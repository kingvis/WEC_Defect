"""Benchmark model families for an operator EARLY-WARNING system, and pick the best.

Clean accuracy is ~100% for every deep model, so that doesn't discriminate. What matters for
predicting/alerting on developing faults is:
  - incipient recall : on the MILDEST faults (PTO sev 0.8, mooring sev 0.75), does it still
                       flag a fault?  (catching the earliest sign = the prognostic value)
  - macro-F1         : balanced 4-class quality
  - noise @10%       : robustness to sensor noise (operator field conditions)
  - size / speed     : light enough to run live on an operator console (CPU)

Compares: CNN, LSTM, GRU, TCN (deep) and RandomForest, LogisticRegression (classical on
engineered features). Run-aware split (no window leakage).

Usage:  py -3.13 -m tools.run_benchmark
"""
from __future__ import annotations
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.datasets import WindowDS
from src.evaluate import predict
from src.features import extract
from src.ingest import parse_meta
from src.models import build_model
from src.train import train
from src.utils import DATA_RAW, OUTPUTS, contiguous_labels, get_logger, label_names, set_seed
from src.windowing import CHANNELS, build_dataset
import glob, os

log = get_logger("benchmark")
MILD = {"pto_damping_loss": 0.8, "mooring_stiffness_loss": 0.75, "pto_plus_mooring": 0.5}


def _per_window_meta(runid):
    files = sorted(glob.glob(str(DATA_RAW / "run_*.csv")))
    fault_of, sev_of = [], []
    for f in files:
        m = parse_meta(f)
        fault_of.append(m["fault"]); sev_of.append(m["severity"])
    fault_of, sev_of = np.array(fault_of, dtype=object), np.array(sev_of)
    return fault_of[runid], sev_of[runid]


def _noise(X, lv, seed=0):
    rng = np.random.default_rng(seed)
    return X + rng.normal(0, lv, X.shape).astype(np.float32) * X.std((0, 1), keepdims=True)


def main():
    set_seed(42)
    X, y_raw, runid = build_dataset(win=384, stride=96, transient=1000)
    y, present = contiguous_labels(y_raw)
    n_cls = len(present)
    healthy_idx = present.index(0)
    fault_w, sev_w = _per_window_meta(runid)

    gss = GroupShuffleSplit(1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups=runid))
    mu, sd = X[tr].mean((0, 1), keepdims=True), X[tr].std((0, 1), keepdims=True) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ytr, yte = y[tr], y[te]
    # incipient test windows = mildest severity of each fault
    inc = np.array([f in MILD and abs(s - MILD[f]) < 1e-6 for f, s in zip(fault_w[te], sev_w[te])])

    def incipient_recall(pred):
        if inc.sum() == 0:
            return float("nan")
        return float((pred[inc] != healthy_idx).mean())   # flagged as *some* fault

    rows = []

    # ---- deep models ----
    for name in ["cnn", "lstm", "gru", "tcn"]:
        m = build_model(name, len(CHANNELS), n_cls)
        nparam = sum(p.numel() for p in m.parameters())
        t0 = time.time()
        # carve a val slice from train for early stopping
        ntr = len(Xtr); idx = np.random.permutation(ntr); cut = int(0.85 * ntr)
        m = train(m, WindowDS(Xtr[idx[:cut]], ytr[idx[:cut]]), WindowDS(Xtr[idx[cut:]], ytr[idx[cut:]]),
                  n_cls=n_cls, y_tr=ytr[idx[:cut]], epochs=20, patience=5)
        ttrain = time.time() - t0
        t1 = time.time(); pred = predict(m, Xte); tinfer = (time.time() - t1) / len(Xte) * 1000
        f1 = f1_score(yte, pred, average="macro")
        accn = (predict(m, _noise(Xte, 0.1)) == yte).mean()
        rows.append((name.upper(), f1, incipient_recall(pred), accn, nparam, ttrain, tinfer))

    # ---- classical on engineered features ----
    log.info("extracting features for classical models ...")
    Ftr, Fte = extract(X[tr]), extract(X[te])
    fs = StandardScaler().fit(Ftr)
    Ftr, Fte = fs.transform(Ftr), fs.transform(Fte)
    for name, clf in [("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
                      ("LogReg", LogisticRegression(max_iter=2000, class_weight="balanced"))]:
        t0 = time.time(); clf.fit(Ftr, ytr); ttrain = time.time() - t0
        pred = clf.predict(Fte)
        f1 = f1_score(yte, pred, average="macro")
        # noise on raw -> re-extract features
        Fn = fs.transform(extract(_noise(X[te], 0.1)))
        accn = (clf.predict(Fn) == yte).mean()
        rows.append((name, f1, incipient_recall(pred), accn, np.nan, ttrain, np.nan))

    # ---- report ----
    print(f"\n{'model':<14}{'macroF1':>9}{'incip.recall':>13}{'acc@10%noise':>14}{'params':>10}{'train(s)':>10}")
    for r in sorted(rows, key=lambda x: (x[2], x[1]), reverse=True):
        p = f"{r[4]:,}" if not np.isnan(r[4]) else "-"
        print(f"{r[0]:<14}{r[1]:>9.3f}{r[2]:>13.3f}{r[3]:>14.3f}{p:>10}{r[5]:>10.1f}")

    best = max(rows, key=lambda x: (round(x[2], 3), round(x[1], 3), round(x[3], 3)))
    print(f"\n>>> BEST for early-warning: {best[0]}  "
          f"(incipient recall {best[2]:.3f}, macroF1 {best[1]:.3f}, noise@10% {best[3]:.3f})")
    _plot(rows)


def _plot(rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    names = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names)); w = 0.27
    ax.bar(x - w, [r[1] for r in rows], w, label="macro-F1")
    ax.bar(x, [r[2] for r in rows], w, label="incipient recall")
    ax.bar(x + w, [r[3] for r in rows], w, label="acc @10% noise")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_title("Model benchmark (early-warning metrics)")
    fig.tight_layout(); out = OUTPUTS / "model_benchmark.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
