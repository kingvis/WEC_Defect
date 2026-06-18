"""End-to-end pipeline: window -> run-aware split -> train -> evaluate -> noise sweep.

Usage:
    python -m tools.run_pipeline --model cnn --epochs 8
    python -m tools.run_pipeline --model lstm --epochs 10

Works on synthetic OR real data — same code, as long as data/raw/ has run_*.csv files.
"""
from __future__ import annotations
import argparse

import numpy as np

from src.datasets import WindowDS, grouped_split_3way
from src.evaluate import (binary_report, evaluate, noise_sweep, plot_confusion,
                          plot_noise, predict)
from src.models import build_model
from src.train import train
from src.utils import contiguous_labels, get_logger, label_names, set_seed
from src.windowing import CHANNELS, build_dataset

log = get_logger("pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--transient", type=int, default=500,
                    help="samples to drop as ramp-up transient (synthetic uses ~500)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # 1. Window all runs (leakage-free: runid kept per window).
    X, y_raw, runid = build_dataset(win=args.win, stride=args.stride, transient=args.transient)
    # Remap sparse semantic labels (0,1,2,4) -> contiguous model indices (0,1,2,3).
    y, present = contiguous_labels(y_raw)
    n_cls = len(present)
    names = label_names()
    target_names = [names.get(orig, f"class{orig}") for orig in present]
    log.info("classes (semantic %s): %s | windows=%d", present, target_names, len(y))

    # 2. Run-aware 3-way split, scaler fit on train only.
    Xtr, ytr, Xval, yval, Xte, yte, (mu, sd) = grouped_split_3way(
        X, y, runid, val_size=0.2, test_size=0.2, seed=args.seed)
    np.savez(_scaler_path(), mu=mu, sd=sd)

    # 3. Train.
    model = build_model(args.model, n_ch=len(CHANNELS), n_cls=n_cls)
    model = train(model, WindowDS(Xtr, ytr), WindowDS(Xval, yval),
                  n_cls=n_cls, y_tr=ytr, epochs=args.epochs)

    # 4. Evaluate: multi-class + collapsed binary fault/no-fault.
    log.info("=== multi-class report (which fault) ===")
    rep, cm = evaluate(model, Xte, yte, target_names=target_names)
    plot_confusion(cm, target_names=target_names)

    log.info("=== binary report (fault vs no-fault) ===")
    pred = predict(model, Xte)
    brep = binary_report(yte, pred)
    print(f"binary fault F1 = {brep['fault']['f1-score']:.3f}")

    # 5. Noise robustness sweep (headline result).
    log.info("=== noise robustness sweep ===")
    curve = noise_sweep(model, Xte, yte)
    plot_noise(curve)

    log.info("DONE. Figures + best.pt in outputs/.")


def _scaler_path():
    from src.utils import OUTPUTS
    return OUTPUTS / "scaler.npz"


if __name__ == "__main__":
    main()
