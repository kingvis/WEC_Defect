"""Leave-one-sea-state-out (LOSO) evaluation — generalization to UNSEEN wave conditions.

The standard split holds out runs but the test sea states still appear in training. LOSO is
harder and more honest: for each of the 9 sea states, train on the other 8 and test on the
held-out one. Reports per-sea-state accuracy and the mean +/- std. This answers "does the
detector work in wave conditions it was never trained on?"

Usage:  py -3.13 -m tools.run_loso
"""
from __future__ import annotations
import glob
import os
import re

import numpy as np

from src.datasets import WindowDS
from src.evaluate import predict
from src.models import build_model
from src.train import train
from src.utils import DATA_RAW, contiguous_labels, get_logger, set_seed
from src.windowing import CHANNELS, build_dataset

log = get_logger("loso")
SS_RE = re.compile(r"_(Hs\d+_Tp\d+)")


def main():
    set_seed(42)
    X, y_raw, runid = build_dataset(win=384, stride=96, transient=1000)
    y, present = contiguous_labels(y_raw)
    n_cls = len(present)

    # Map each runid (sorted-file index) -> its sea state string from the filename.
    files = sorted(glob.glob(str(DATA_RAW / "run_*.csv")))
    ss_of_run = []
    for f in files:
        m = SS_RE.search(os.path.basename(f))
        ss_of_run.append(m.group(1) if m else "NA")
    ss_of_run = np.array(ss_of_run)
    seastate = ss_of_run[runid]                      # per-window sea state

    states = sorted(set(seastate))
    log.info("LOSO over %d sea states: %s", len(states), states)

    accs = []
    for s in states:
        te = seastate == s
        tr = ~te
        mu = X[tr].mean((0, 1), keepdims=True)
        sd = X[tr].std((0, 1), keepdims=True) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        # small val slice from train for early stopping
        n = len(Xtr); idx = np.random.permutation(n); cut = int(0.85 * n)
        tri, vai = idx[:cut], idx[cut:]
        model = train(build_model("cnn", len(CHANNELS), n_cls),
                      WindowDS(Xtr[tri], y[tr][tri]), WindowDS(Xtr[vai], y[tr][vai]),
                      n_cls=n_cls, y_tr=y[tr][tri], epochs=20, patience=5)
        acc = float((predict(model, Xte) == y[te]).mean())
        accs.append(acc)
        log.info("held-out %s : test acc = %.3f  (n=%d windows)", s, acc, te.sum())

    accs = np.array(accs)
    print("\n=== Leave-one-sea-state-out summary ===")
    for s, a in zip(states, accs):
        print(f"  {s:<18} {a:.3f}")
    print(f"  {'MEAN +/- STD':<18} {accs.mean():.3f} +/- {accs.std():.3f}")


if __name__ == "__main__":
    main()
