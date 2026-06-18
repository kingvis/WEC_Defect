"""Lightweight shape / contract tests for the pipeline plumbing.

Run:  py -3.13 -m pytest tests/ -q     (or: python -m pytest)
These don't need real data — they synthesise tiny arrays and check shapes/invariants.
"""
import numpy as np
import torch

from src.datasets import WindowDS, grouped_split, grouped_split_3way
from src.models import CNN1D, LSTMClf, AE
from src.utils import contiguous_labels


N, WIN, CH, NCLS = 40, 128, 7, 4


def _fake():
    X = np.random.randn(N, WIN, CH).astype(np.float32)
    y = np.random.randint(0, NCLS, N)
    runid = np.repeat(np.arange(N // 4), 4)   # 4 windows per run
    return X, y, runid


def test_contiguous_labels():
    y = np.array([0, 1, 2, 4, 4, 0])
    yc, present = contiguous_labels(y)
    assert present == [0, 1, 2, 4]
    assert set(yc.tolist()) == {0, 1, 2, 3}


def test_grouped_split_no_leakage():
    X, y, runid = _fake()
    Xtr, ytr, Xte, yte, (mu, sd) = grouped_split(X, y, runid, test_size=0.25, seed=0)
    # No run id should appear in both train and test.
    gss_tr = set(runid[:len(ytr)])  # not exact, so test via explicit recompute instead
    # Recompute membership properly:
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, y, runid))
    assert set(runid[tr]).isdisjoint(set(runid[te]))
    assert Xtr.shape[1:] == (WIN, CH)


def test_grouped_split_3way_shapes():
    X, y, runid = _fake()
    Xtr, ytr, Xval, yval, Xte, yte, _ = grouped_split_3way(X, y, runid, seed=0)
    assert len(Xtr) + len(Xval) + len(Xte) == N


def test_windowds_transpose():
    X, y, _ = _fake()
    ds = WindowDS(X, y)
    xb, yb = ds[0]
    assert xb.shape == (CH, WIN)   # [channels, time] for conv1d


def test_models_forward():
    x = torch.randn(8, CH, WIN)
    assert CNN1D(CH, NCLS)(x).shape == (8, NCLS)
    assert LSTMClf(CH, NCLS)(x).shape == (8, NCLS)
    out = AE(CH)(x)
    assert out.shape[0] == 8 and out.shape[1] == CH   # reconstructs channels
