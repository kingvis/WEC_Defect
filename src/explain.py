"""SHAP explainability tied to failure physics (build guide §6.9).

Aggregate |SHAP| per channel to see which signal reveals each fault, then map to physics:
  mooring fault -> Tmoor + low-frequency surge drift;  PTO fault -> Ppto / vrel phase change.
"""
from __future__ import annotations

import numpy as np
import torch

from .utils import OUTPUTS, get_logger
from .windowing import CHANNELS

log = get_logger("explain")


def shap_channel_importance(model, X_background, X_explain, n_bg=100, n_explain=200):
    """Return mean |SHAP| per channel, aggregated over time and samples.

    Uses GradientExplainer (works for torch models). Background kept small for CPU speed.
    """
    import shap

    bg = torch.tensor(np.asarray(X_background[:n_bg]), dtype=torch.float32).transpose(1, 2)
    ex = torch.tensor(np.asarray(X_explain[:n_explain]), dtype=torch.float32).transpose(1, 2)

    explainer = shap.GradientExplainer(model, bg)
    sv = explainer.shap_values(ex)            # list (per class) of [n, ch, time], or array

    # Normalise to array [classes, n, ch, time]
    arr = np.stack(sv) if isinstance(sv, list) else sv[None]
    # mean |SHAP| over samples and time -> [classes, ch]; then average classes -> [ch]
    per_class_ch = np.abs(arr).mean(axis=(1, 3))      # [classes, ch]
    overall = per_class_ch.mean(axis=0)               # [ch]
    return dict(zip(CHANNELS, overall.tolist())), per_class_ch


def map_to_physics(channel_importance: dict) -> str:
    """Heuristic narrative linking the top channels to expected failure physics."""
    ranked = sorted(channel_importance.items(), key=lambda kv: kv[1], reverse=True)
    top = [c for c, _ in ranked[:3]]
    notes = []
    if "Tmoor" in top:
        notes.append("mooring tension dominant -> consistent with mooring stiffness loss / drift")
    if "Ppto" in top or "vrel" in top or "Fpto" in top:
        notes.append("PTO power/velocity dominant -> consistent with PTO damping loss")
    return f"Top channels: {top}. " + "; ".join(notes)


def plot_channel_importance(channel_importance: dict, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out = out or (OUTPUTS / "shap_channel_importance.png")
    chs = list(channel_importance)
    vals = [channel_importance[c] for c in chs]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(chs, vals)
    ax.set_ylabel("mean |SHAP|"); ax.set_title("Channel importance")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("saved %s", out)
