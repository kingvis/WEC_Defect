"""Torch Dataset + run-aware splits with leakage-free scaling.

Split BY RUN (GroupShuffleSplit on runid) and fit the scaler on TRAIN windows only.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


class WindowDS(Dataset):
    """Windows as [N, channels, time] tensors for 1D-CNN / LSTM."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X arrives [N, time, channels] -> transpose to [N, channels, time]
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def grouped_split(X, y, runid, test_size: float = 0.25, seed: int = 42):
    """Run-aware train/test split + train-only standardisation.

    Returns Xtr, ytr, Xte, yte, (mu, sd). Scaler stats are computed on train windows only.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups=runid))
    mu = X[tr].mean((0, 1), keepdims=True)
    sd = X[tr].std((0, 1), keepdims=True) + 1e-8
    Xtr = (X[tr] - mu) / sd
    Xte = (X[te] - mu) / sd
    return Xtr, y[tr], Xte, y[te], (mu, sd)


def grouped_split_3way(X, y, runid, val_size=0.2, test_size=0.2, seed=42):
    """Train/val/test split, all run-aware, scaler fit on train only."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval, te = next(gss1.split(X, y, groups=runid))
    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr_rel, val_rel = next(gss2.split(X[trval], y[trval], groups=runid[trval]))
    tr, val = trval[tr_rel], trval[val_rel]

    mu = X[tr].mean((0, 1), keepdims=True)
    sd = X[tr].std((0, 1), keepdims=True) + 1e-8
    scale = lambda a: (a - mu) / sd
    return (scale(X[tr]), y[tr], scale(X[val]), y[val], scale(X[te]), y[te], (mu, sd))
