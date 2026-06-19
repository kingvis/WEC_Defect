"""Unsupervised unseen-fault detection (build guide §6.7).

Train a conv autoencoder on HEALTHY windows ONLY. A fault type the model never saw should
reconstruct poorly -> high error -> flagged as anomaly. We report ROC-AUC of healthy(test)
vs each fault type, plus the detection rate at a 3-sigma healthy threshold.

This is the "detect faults you never trained on" contribution — distinct from the supervised
classifier, which needs labelled examples of every fault.

Usage:  py -3.13 -m tools.run_anomaly
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.anomaly import evaluate_unseen, fit_threshold, recon_error, train_ae
from src.utils import get_logger, label_names, set_seed
from src.windowing import CHANNELS, build_dataset

log = get_logger("anomaly")


def main():
    set_seed(42)
    X, y, runid = build_dataset(win=384, stride=96, transient=1000)
    names = label_names()
    HEALTHY = 0

    # Standardise on healthy-train stats only (the AE only ever sees healthy at fit time).
    heal_mask = y == HEALTHY
    heal_runs = runid[heal_mask]
    # split healthy RUNS into AE-train vs healthy-test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    tr_idx, te_idx = next(gss.split(X[heal_mask], y[heal_mask], groups=heal_runs))
    Xh = X[heal_mask]
    Xh_tr, Xh_te = Xh[tr_idx], Xh[te_idx]

    mu = Xh_tr.mean((0, 1), keepdims=True)
    sd = Xh_tr.std((0, 1), keepdims=True) + 1e-8
    scale = lambda a: (a - mu) / sd

    ae = train_ae(scale(Xh_tr), n_ch=len(CHANNELS), epochs=40)

    err_h_tr = recon_error(ae, scale(Xh_tr))
    thr = fit_threshold(err_h_tr, k=3.0)
    err_h_te = recon_error(ae, scale(Xh_te))

    log.info("=== unseen-fault detection (AE trained on HEALTHY only) ===")
    log.info("healthy-test mean recon err = %.4f (threshold = %.4f)", err_h_te.mean(), thr)
    print(f"\n{'held-out fault':<26}{'ROC-AUC':>9}{'detected@3sigma':>18}")
    print(f"{'healthy (reference)':<26}{'-':>9}{(err_h_te > thr).mean():>17.0%}")
    for lbl in sorted(set(y) - {HEALTHY}):
        Xf = X[y == lbl]
        auc, _, err_f = evaluate_unseen(ae, scale(Xh_te), scale(Xf))
        det = (err_f > thr).mean()
        print(f"{names.get(lbl, str(lbl)):<26}{auc:>9.3f}{det:>17.0%}")


if __name__ == "__main__":
    main()
