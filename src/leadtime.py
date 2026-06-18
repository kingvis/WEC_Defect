"""Early / incipient detection with lead-time quantification (build guide §6.8).

Slide a trained classifier along a DEGRADATION-TRAJECTORY run (severity worsens over time)
and record when it first confidently flags the fault. Lead time = t(functional failure) -
t(first confident detection). This quantified curve is a novel deliverable.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from .windowing import CHANNELS


def slide_windows(df, win=512, stride=64, transient=2000):
    """Yield (center_time, window) along a single degradation run."""
    t = df["t"].to_numpy()
    x = df[CHANNELS].to_numpy(dtype=np.float32)
    for s in range(transient, len(x) - win, stride):
        yield float(t[s + win // 2]), x[s:s + win]


def detect_trajectory(model, df, mu, sd, win=512, stride=64, transient=2000,
                      fault_prob_thresh=0.7, persistence=3, device="cpu"):
    """Find the first time the fault is flagged confidently for `persistence` windows in a row.

    Returns dict with detect_time, prob trace, and the per-window times. Healthy = label 0;
    "fault probability" = 1 - P(healthy).
    """
    model.eval()
    times, probs = [], []
    streak, detect_time = 0, None
    for tc, w in slide_windows(df, win, stride, transient):
        wn = (w - mu[0]) / sd[0]                      # apply train scaler
        xt = torch.tensor(wn, dtype=torch.float32).T.unsqueeze(0).to(device)  # [1, ch, time]
        with torch.no_grad():
            p = torch.softmax(model(xt), dim=1).cpu().numpy()[0]
        p_fault = 1.0 - p[0]
        times.append(tc); probs.append(p_fault)
        if p_fault >= fault_prob_thresh:
            streak += 1
            if streak >= persistence and detect_time is None:
                detect_time = times[-persistence]     # first window of the confident streak
        else:
            streak = 0
    return {"detect_time": detect_time,
            "times": np.array(times), "p_fault": np.array(probs)}


def lead_time(detect_time: Optional[float], failure_time: float) -> Optional[float]:
    """Lead time in seconds: how early the fault was caught before functional failure."""
    if detect_time is None:
        return None
    return float(failure_time - detect_time)
