"""Optional engineered features (statistical + spectral) per window.

Two uses: (1) classical-ML baselines (RandomForest etc.) over tabular features, and
(2) sanity-checking class separability before training deep models (Gate 2).
Deep models (CNN/LSTM) consume raw windows directly and do NOT need these.
"""
from __future__ import annotations
from typing import List

import numpy as np
from scipy import signal, stats

from .windowing import CHANNELS


def _stat_features(w: np.ndarray) -> List[float]:
    """Per-channel time-domain statistics for one [win, n_channels] window."""
    feats = []
    for c in range(w.shape[1]):
        x = w[:, c]
        feats += [
            float(np.mean(x)), float(np.std(x)),
            float(np.min(x)), float(np.max(x)),
            float(stats.skew(x)), float(stats.kurtosis(x)),
            float(np.sqrt(np.mean(x ** 2))),                 # RMS
            float(np.mean(np.abs(np.diff(x)))),              # mean abs change
        ]
    return feats


def _spectral_features(w: np.ndarray, fs: float = 10.0) -> List[float]:
    """Per-channel spectral summary: dominant freq, spectral centroid, band power."""
    feats = []
    for c in range(w.shape[1]):
        f, pxx = signal.welch(w[:, c], fs=fs, nperseg=min(256, w.shape[0]))
        psum = pxx.sum() + 1e-12
        feats += [
            float(f[np.argmax(pxx)]),               # dominant frequency
            float((f * pxx).sum() / psum),          # spectral centroid
            float(pxx[f < 0.1].sum() / psum),       # low-freq band fraction (mooring drift)
            float(pxx[f > 0.3].sum() / psum),       # high-freq band fraction
        ]
    return feats


def feature_names(fs: float = 10.0) -> List[str]:
    names = []
    stat = ["mean", "std", "min", "max", "skew", "kurt", "rms", "mac"]
    spec = ["fdom", "fcentroid", "lowband", "highband"]
    for ch in CHANNELS:
        names += [f"{ch}_{s}" for s in stat]
    for ch in CHANNELS:
        names += [f"{ch}_{s}" for s in spec]
    return names


def extract(X: np.ndarray, fs: float = 10.0) -> np.ndarray:
    """[N, win, n_channels] -> [N, n_features] feature matrix."""
    rows = [(_stat_features(w) + _spectral_features(w, fs)) for w in X]
    return np.asarray(rows, dtype=np.float32)
