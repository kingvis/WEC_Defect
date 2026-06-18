"""Derive an Hs/Tp sea-state matrix from NDBC standard meteorological buoy data.

Run:  python -m src.sea_states
Writes config/sea_states.yaml. Falls back gracefully if NDBC is unreachable.
"""
from __future__ import annotations
import io

import numpy as np
import pandas as pd
import requests
import yaml

from .utils import CONFIG_DIR, get_logger

log = get_logger("sea_states")


def fetch_ndbc_year(station: str, year: int) -> pd.DataFrame:
    """NDBC historical standard met file (yearly).

    Returns a dataframe with WVHT (significant wave height Hs) and DPD (dominant period).
    'MM'/99/999 are NDBC missing-value codes.
    """
    url = (f"https://www.ndbc.noaa.gov/view_text_file.php?"
           f"filename={station}h{year}.txt.gz&dir=data/historical/stdmet/")
    raw = requests.get(url, timeout=60).text
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", skiprows=[1],
                     na_values=["MM", 99, 999, 9999])
    return df


def build_sea_state_matrix(stations, years, n_hs=8, n_tp=5, out=None):
    """Build and write a defensible Hs/Tp grid (NREL-style: Hs 0.25-3.75 m, Tp 5-13 s)."""
    out = out or (CONFIG_DIR / "sea_states.yaml")
    frames = []
    for s in stations:
        for y in years:
            try:
                frames.append(fetch_ndbc_year(s, y)[["WVHT", "DPD"]])
                log.info("fetched %s %s", s, y)
            except Exception as e:  # network / missing file / parse
                log.warning("skip %s%s: %s", s, y, e)

    if not frames:
        log.error("No NDBC data fetched; keeping existing %s", out)
        return None

    d = pd.concat(frames).dropna()
    d = d[(d.WVHT > 0) & (d.WVHT < 12) & (d.DPD > 2) & (d.DPD < 25)]
    hs = np.round(np.linspace(max(0.25, d.WVHT.quantile(.05)),
                              min(3.75, d.WVHT.quantile(.95)), n_hs), 2)
    tp = np.round(np.linspace(d.DPD.quantile(.10), d.DPD.quantile(.90), n_tp), 1)
    grid = [{"Hs": float(h), "Tp": float(t)} for h in hs for t in tp]
    with open(out, "w") as f:
        yaml.safe_dump({"sea_states": grid}, f, sort_keys=False)
    log.info("%d sea states written to %s", len(grid), out)
    return grid


if __name__ == "__main__":
    # Example Pacific buoys; adjust to buoys near the intended deployment region.
    build_sea_state_matrix(["46022", "46050"], [2018, 2019])
