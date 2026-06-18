"""Load raw per-run CSVs exported from WEC-Sim, with basic cleaning + metadata parsing."""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .utils import DATA_RAW

# Metadata encoded in the filename, e.g.
#   run_00042__pto_damping_loss_sev60_Hs150_Tp080.csv
FNAME_RE = re.compile(
    r"run_(?P<runid>\d+)__(?P<fault>[a-z_]+)_sev(?P<sev>\d+)_Hs(?P<hs>\d+)_Tp(?P<tp>\d+)"
)


def parse_meta(path: str | Path) -> Dict[str, str]:
    """Parse fault / severity / sea-state out of a run filename."""
    m = FNAME_RE.search(os.path.basename(str(path)))
    if not m:
        return {}
    d = m.groupdict()
    return {
        "runid": int(d["runid"]),
        "fault": d["fault"],
        "severity": int(d["sev"]) / 100.0,
        "Hs": int(d["hs"]) / 100.0,
        "Tp": int(d["tp"]) / 10.0,
    }


def load_run(path: str | Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load one run CSV; drop fully-NaN rows; forward/back-fill small gaps."""
    df = pd.read_csv(path)
    df = df.dropna(how="all").ffill().bfill()
    return df, parse_meta(path)


def manifest(raw_dir: str | Path = DATA_RAW) -> pd.DataFrame:
    """Build a manifest of all runs in data/raw/ from their filenames."""
    rows = []
    for p in sorted(Path(raw_dir).glob("run_*.csv")):
        meta = parse_meta(p)
        if meta:
            meta["file"] = str(p)
            rows.append(meta)
    return pd.DataFrame(rows)
