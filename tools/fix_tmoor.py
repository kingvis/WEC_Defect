"""Reconstruct the mooring-tension channel (Tmoor) in already-exported CSVs.

The first unified MATLAB export had a bug (isfield on an object) that wrote Tmoor=0.
For a LINEAR mooring matrix the surge tension is exactly  F = -K_surge * x  (Hooke's law),
and the smoke test confirmed F_std (70122) ~= K(1e5) * float-surge-std(0.68). So we rebuild
Tmoor = -K * x1, with K read from the fault/severity encoded in each filename.

This is the same quantity WEC-Sim computes; it just got zeroed on export. For the FINAL
released dataset, regenerate with the fixed matlab/generate_unified_dataset.m for authentic output.

Usage:  py -3.13 -m tools.fix_tmoor
"""
from __future__ import annotations
import glob

import pandas as pd

from src.ingest import parse_meta
from src.utils import DATA_RAW, get_logger

log = get_logger("fix_tmoor")

K_NOMINAL = 1.0e5   # nominal mooring surge stiffness [N/m] (matches the generator)


def mooring_stiffness(fault: str, severity: float) -> float:
    """Mooring surge stiffness for a run, from its fault label + severity."""
    if fault in ("mooring_stiffness_loss", "pto_plus_mooring"):
        return K_NOMINAL * severity      # degraded line -> reduced stiffness
    return K_NOMINAL                      # healthy / pto: nominal mooring


def main():
    paths = sorted(glob.glob(str(DATA_RAW / "run_*.csv")))
    if not paths:
        raise SystemExit("No CSVs in data/raw/ — extract the dataset first.")
    fixed = 0
    for p in paths:
        meta = parse_meta(p)
        if not meta:
            continue
        df = pd.read_csv(p)
        if "x1" not in df.columns:
            log.warning("no x1 column, skipping %s", p)
            continue
        K = mooring_stiffness(meta["fault"], meta["severity"])
        df["Tmoor"] = -K * df["x1"]       # linear mooring restoring force in surge
        df.to_csv(p, index=False)
        fixed += 1
    log.info("reconstructed Tmoor in %d files (K_nominal=%.0f)", fixed, K_NOMINAL)


if __name__ == "__main__":
    main()
