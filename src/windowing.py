"""Segment per-run signals into fixed windows WITHOUT leakage.

Leakage rule (critical, build guide §6.2): split BY RUN, not by window. Every window from a
single simulation must fall entirely in train OR test. This module only produces windows +
a run-id per window; the run-aware split lives in datasets.py.
"""
from __future__ import annotations
import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .ingest import load_run
from .utils import DATA_PROCESSED, get_logger, label_map

log = get_logger("windowing")

# 9 channels logged per run. Heave-side (PTO fault) + surge-side (mooring fault).
#   z1,z2 heave; rel,vrel relative-heave + PTO velocity; Fpto,Ppto PTO force/power;
#   x1,x2 surge (mooring drift); Tmoor mooring surge force.
CHANNELS = ["z1", "z2", "rel", "vrel", "Fpto", "Ppto", "x1", "x2", "Tmoor"]


def window_run(df, win: int = 512, stride: int = 256, transient: int = 2000) -> np.ndarray:
    """Slice one run into [n_windows, win, n_channels]. Drops the initial ramp-up transient."""
    x = df[CHANNELS].to_numpy(dtype=np.float32)
    x = x[transient:]                      # discard ramp-up transient (tune to your dt)
    out = []
    for s in range(0, len(x) - win, stride):
        out.append(x[s:s + win])
    if not out:
        return np.empty((0, win, len(CHANNELS)), np.float32)
    return np.stack(out)


def build_dataset(raw_glob: str = None, win: int = 512, stride: int = 256,
                  transient: int = 2000, label_map_: Optional[Dict[str, int]] = None,
                  out: str | Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window every run in data/raw/ and save a compressed .npz of (X, y, runid).

    X     : [N, win, n_channels] float32
    y     : [N] int label
    runid : [N] int — group id used for run-aware splitting
    """
    from .utils import DATA_RAW
    raw_glob = raw_glob or str(DATA_RAW / "run_*.csv")
    out = out or (DATA_PROCESSED / "dataset.npz")
    label_map_ = label_map_ or label_map()

    X, y, runid = [], [], []
    for i, path in enumerate(sorted(glob.glob(raw_glob))):
        df, meta = load_run(path)
        if not meta or meta["fault"] not in label_map_:
            log.warning("skip (no label) %s", path)
            continue
        w = window_run(df, win, stride, transient)
        if len(w) == 0:
            log.warning("skip (too short) %s", path)
            continue
        lbl = label_map_[meta["fault"]]
        X.append(w)
        y.append(np.full(len(w), lbl))
        runid.append(np.full(len(w), i))

    if not X:
        raise RuntimeError(f"No windows built from {raw_glob}. Is data/raw/ populated?")

    X = np.concatenate(X)
    y = np.concatenate(y)
    runid = np.concatenate(runid)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, runid=runid)
    log.info("dataset: X=%s y=%s runs=%d -> %s",
             X.shape, y.shape, len(np.unique(runid)), out)
    return X, y, runid


def load_processed(path: str | Path = None):
    path = path or (DATA_PROCESSED / "dataset.npz")
    d = np.load(path)
    return d["X"], d["y"], d["runid"]
