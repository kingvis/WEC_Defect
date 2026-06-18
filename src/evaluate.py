"""Evaluation: classification report, confusion matrix, and the noise-robustness sweep."""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .utils import OUTPUTS, get_logger

log = get_logger("evaluate")


def predict(model, X, device="cpu", bs=256) -> np.ndarray:
    """Batched argmax prediction over [N, time, channels] input."""
    model.eval()
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32).transpose(1, 2)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            preds.append(model(Xt[i:i + bs].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate(model, X, y, target_names=None, device="cpu"):
    """Print a per-class report; return (report_dict, confusion_matrix)."""
    pred = predict(model, X, device)
    rep = classification_report(y, pred, digits=3, zero_division=0,
                                target_names=target_names, output_dict=True)
    print(classification_report(y, pred, digits=3, zero_division=0,
                                target_names=target_names))
    cm = confusion_matrix(y, pred)
    return rep, cm


def binary_report(y_true, y_pred_multiclass):
    """Collapse multi-class predictions to fault/no-fault (label 0 = healthy)."""
    yt = (np.asarray(y_true) != 0).astype(int)
    yp = (np.asarray(y_pred_multiclass) != 0).astype(int)
    return classification_report(yt, yp, digits=3, zero_division=0,
                                 target_names=["healthy", "fault"], output_dict=True)


def noise_sweep(model, X, y, levels=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2),
                device="cpu", seed=0) -> Dict[float, float]:
    """Add Gaussian noise of increasing std (relative to per-channel std); accuracy curve.

    Headline robustness result (build guide §6.6) — report this as a curve, not an afterthought.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    chan_std = X.std((0, 1), keepdims=True)
    out = {}
    for lv in levels:
        Xn = X + rng.normal(0, lv, X.shape).astype(np.float32) * chan_std
        pred = predict(model, Xn, device)
        out[float(lv)] = float((pred == y).mean())
        log.info("noise %.3f -> acc %.3f", lv, out[float(lv)])
    return out


def plot_confusion(cm, target_names=None, out=None):
    """Save a confusion-matrix figure to outputs/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or (OUTPUTS / "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ticks = range(len(cm))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    if target_names:
        ax.set_xticklabels(target_names, rotation=45, ha="right")
        ax.set_yticklabels(target_names)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("saved %s", out)


def plot_noise(curve: Dict[float, float], out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or (OUTPUTS / "noise_robustness.png")
    lv = sorted(curve)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lv, [curve[k] for k in lv], "o-")
    ax.set_xlabel("relative noise std"); ax.set_ylabel("accuracy")
    ax.set_title("Noise robustness"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("saved %s", out)
