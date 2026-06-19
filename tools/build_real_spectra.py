"""Fetch real NDBC spectral-wave-density (swden) data and select a representative spread of
measured spectra (calm -> storm) for WEC-Sim spectrumImport.

Output: data/ndbc/real_spectra.mat  (upload this ONE file to MATLAB Online /MATLAB Drive)
  contains: freqs [Hz] (M,), S_all [N,M] (m^2/Hz), Hs [N], Tp [N], src {N}
WEC-Sim spectrumImport wants a 2-col [freq_Hz, S] file; the generator writes each row of
S_all to a temp file and points waves.spectrumFile at it.

Usage:  py -3.13 -m tools.build_real_spectra
"""
from __future__ import annotations
import io

import numpy as np
import pandas as pd
import requests
from scipy.io import savemat

from src.utils import DATA_RAW, get_logger

log = get_logger("spectra")

BUOYS = ["46022", "46050", "46001"]      # moderate Pacific + Gulf of Alaska (storms)
YEARS = [2021, 2022]
TARGET_HS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]   # span calm -> storm
COMMON_F = np.round(np.arange(0.035, 0.485, 0.005), 3)       # common freq grid [Hz]


def fetch_swden(buoy, year):
    url = (f"https://www.ndbc.noaa.gov/view_text_file.php?"
           f"filename={buoy}w{year}.txt.gz&dir=data/historical/swden/")
    txt = requests.get(url, timeout=60, verify=False).text
    lines = txt.splitlines()
    freqs = np.array([float(x) for x in lines[0].split()[5:]])
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+", skiprows=1, header=None)
    S = df.iloc[:, 5:].to_numpy(float)
    S[S >= 999.0] = np.nan
    ok = ~np.isnan(S).any(axis=1)
    return freqs, S[ok]


def main():
    import urllib3
    urllib3.disable_warnings()
    recs = []     # (Hs, Tp, S_on_common_grid, src)
    for b in BUOYS:
        for y in YEARS:
            try:
                f, S = fetch_swden(b, y)
            except Exception as e:
                log.warning("skip %s %s: %s", b, y, e); continue
            Si = np.array([np.interp(COMMON_F, f, s) for s in S])     # to common grid
            m0 = np.trapz(Si, COMMON_F, axis=1)
            Hs = 4.0 * np.sqrt(np.clip(m0, 0, None))
            fp = COMMON_F[np.argmax(Si, axis=1)]
            Tp = 1.0 / np.clip(fp, 1e-3, None)
            for i in range(len(Si)):
                if 0.3 < Hs[i] < 12 and 4 < Tp[i] < 25:
                    recs.append((Hs[i], Tp[i], Si[i], f"{b}_{y}"))
            log.info("%s %s: %d valid spectra", b, y, len(Si))

    Hs_all = np.array([r[0] for r in recs])
    log.info("pool: %d spectra, Hs P50=%.1f P90=%.1f P99=%.1f max=%.1f",
             len(recs), np.percentile(Hs_all, 50), np.percentile(Hs_all, 90),
             np.percentile(Hs_all, 99), Hs_all.max())

    # pick the spectrum closest to each target Hs
    chosen = []
    for tgt in TARGET_HS:
        j = int(np.argmin(np.abs(Hs_all - tgt)))
        chosen.append(j)
    sel = [recs[j] for j in chosen]

    out = DATA_RAW / "real_spectra.mat"
    savemat(out, {
        "freqs": COMMON_F.reshape(-1, 1),
        "S_all": np.array([s[2] for s in sel]),
        "Hs": np.array([s[0] for s in sel]).reshape(-1, 1),
        "Tp": np.array([s[1] for s in sel]).reshape(-1, 1),
        "src": np.array([s[3] for s in sel], dtype=object).reshape(-1, 1),
    })
    print(f"\nselected {len(sel)} real spectra -> {out}")
    print(f"{'#':>2} {'Hs(m)':>6} {'Tp(s)':>6}  source")
    for i, s in enumerate(sel, 1):
        print(f"{i:>2} {s[0]:>6.2f} {s[1]:>6.1f}  {s[3]}")


if __name__ == "__main__":
    main()
