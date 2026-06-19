"""Explainability: per-class channel importance via occlusion, mapped to fault physics.

For each input channel, zero it out (channels are standardised, so 0 = its mean) and measure
the drop in per-class accuracy. A large drop for class c when channel j is removed means the
model relies on channel j to recognise fault c. We then check this against expected physics:
  mooring fault  -> Tmoor (mooring tension) + surge (x1,x2)
  PTO fault      -> Ppto / Fpto / vrel (power-train signals)

Usage:  py -3.13 -m tools.run_explain
"""
from __future__ import annotations

import numpy as np

from src.datasets import WindowDS, grouped_split_3way
from src.evaluate import predict
from src.models import build_model
from src.train import train
from src.utils import OUTPUTS, get_logger, label_names, set_seed
from src.windowing import CHANNELS, build_dataset

log = get_logger("explain")


def main():
    set_seed(42)
    X, y_raw, runid = build_dataset(win=384, stride=96, transient=1000)
    from src.utils import contiguous_labels
    y, present = contiguous_labels(y_raw)
    n_cls = len(present)
    names = label_names()
    target = [names.get(o, f"c{o}") for o in present]

    Xtr, ytr, Xval, yval, Xte, yte, _ = grouped_split_3way(X, y, runid, seed=42)
    model = train(build_model("cnn", len(CHANNELS), n_cls), WindowDS(Xtr, ytr),
                  WindowDS(Xval, yval), n_cls=n_cls, y_tr=ytr, epochs=25)

    base = predict(model, Xte)
    base_acc = {c: (base[yte == c] == c).mean() for c in range(n_cls)}

    # importance[c, j] = drop in class-c accuracy when channel j is occluded
    imp = np.zeros((n_cls, len(CHANNELS)))
    for j in range(len(CHANNELS)):
        Xocc = Xte.copy()
        Xocc[:, :, j] = 0.0
        pj = predict(model, Xocc)
        for c in range(n_cls):
            m = yte == c
            if m.any():
                imp[c, j] = base_acc[c] - (pj[m] == c).mean()

    # Report
    log.info("=== per-class channel importance (accuracy drop when channel removed) ===")
    print(f"{'class':<24}" + "".join(f"{ch:>7}" for ch in CHANNELS))
    for c in range(n_cls):
        print(f"{target[c]:<24}" + "".join(f"{imp[c, j]:>7.2f}" for j in range(len(CHANNELS))))

    print("\n--- top channels per fault (physics check) ---")
    for c in range(n_cls):
        order = np.argsort(imp[c])[::-1]
        tops = [CHANNELS[j] for j in order[:3] if imp[c, order[list(order).index(j)]] > 0.01]
        print(f"{target[c]:<24} -> {tops}")

    _plot(imp, target, CHANNELS)


def _plot(imp, target, channels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = OUTPUTS / "channel_importance.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(imp, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right")
    ax.set_yticks(range(len(target)))
    ax.set_yticklabels(target)
    ax.set_title("Per-class channel importance (occlusion)")
    for i in range(len(target)):
        for j in range(len(channels)):
            ax.text(j, i, f"{imp[i, j]:.2f}", ha="center", va="center",
                    color="white" if imp[i, j] < imp.max() * 0.6 else "black", fontsize=7)
    fig.colorbar(im, label="accuracy drop")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
