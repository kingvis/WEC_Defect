"""Generate synthetic RM3-like run CSVs so the Python pipeline is testable BEFORE WEC-Sim.

This is a stand-in for WEC-Sim output — NOT a physics model and NOT for thesis results.
It produces CSVs with the exact schema/filenames the real pipeline expects, with
fault-dependent signal changes so the classifier has something real to learn. Once real
WEC-Sim CSVs land in data/raw/, delete these and rerun the same pipeline unchanged.

Usage:  python -m tools.make_synthetic_data --runs-per-class 6
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from src.utils import DATA_RAW, ensure_dirs, get_logger, load_faults

log = get_logger("synth")

# Sea states to sample (Hs, Tp); mirrors config/sea_states.yaml starter grid.
SEA_STATES = [(0.75, 6.0), (0.75, 9.0), (1.75, 9.0), (1.75, 12.0), (3.0, 6.0), (3.0, 12.0)]

DT = 0.1
END_T = 400.0
NOMINAL_PTO_C = 1.2e6
NOMINAL_MOOR_K = 1.0e4


def _irregular_heave(t, Hs, Tp, rng, n_comp=20):
    """Sum of wave components around the peak period -> realistic-ish relative heave."""
    fp = 1.0 / Tp
    freqs = np.linspace(0.4 * fp, 2.5 * fp, n_comp)
    amps = np.exp(-((freqs - fp) ** 2) / (2 * (0.15 * fp) ** 2))
    amps = amps / amps.sum() * (Hs / 2.0)
    phases = rng.uniform(0, 2 * np.pi, n_comp)
    return sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(amps, freqs, phases))


def synth_run(fault: str, sev: float, Hs: float, Tp: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, END_T, DT)

    pto_c = NOMINAL_PTO_C
    moor_k = NOMINAL_MOOR_K
    drift = np.zeros_like(t)

    # Fault-dependent parameter changes.
    if fault in ("pto_damping_loss", "pto_plus_mooring"):
        pto_c *= sev                              # weaker damping -> larger, less-damped motion
    if fault in ("mooring_stiffness_loss", "pto_plus_mooring"):
        moor_k *= sev
        # softer mooring -> slow low-frequency drift (the physics SHAP should reveal)
        drift = 0.4 * (1 - sev) * np.sin(2 * np.pi * 0.02 * t + rng.uniform(0, 6.28))

    # Relative heave (drives PTO). Weaker PTO damping -> amplified motion.
    amp_gain = (NOMINAL_PTO_C / pto_c) ** 0.5
    rel = amp_gain * _irregular_heave(t, Hs, Tp, rng) + drift
    vrel = np.gradient(rel, DT)

    # Float / spar heave (rel = z1 - z2); split with a shared common-mode component.
    common = 0.3 * _irregular_heave(t, Hs, Tp * 1.1, rng)
    z1 = common + rel / 2
    z2 = common - rel / 2

    Fpto = pto_c * vrel
    Ppto = Fpto * vrel                            # instantaneous PTO power

    # Surge side (mooring): softer mooring -> larger low-frequency surge drift.
    surge_gain = (NOMINAL_MOOR_K / moor_k) ** 0.5
    x_common = surge_gain * 0.5 * _irregular_heave(t, Hs, Tp * 1.3, rng) + drift
    x1 = x_common
    x2 = x_common * 0.9
    Tmoor = moor_k * x1 + rng.normal(0, 0.01 * moor_k, len(t))   # mooring surge force

    # Light sensor noise on every channel.
    def noisy(x, frac=0.01):
        return x + rng.normal(0, frac * (np.std(x) + 1e-9), len(x))

    return pd.DataFrame({
        "t": t, "z1": noisy(z1), "z2": noisy(z2), "rel": noisy(rel), "vrel": noisy(vrel),
        "Fpto": noisy(Fpto), "Ppto": noisy(Ppto),
        "x1": noisy(x1), "x2": noisy(x2), "Tmoor": noisy(Tmoor),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-per-class", type=int, default=6,
                    help="distinct (severity, sea-state) runs per fault class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dirs()
    cfg = load_faults()
    # Build a (fault, severity_levels, label) list from config (data-driven).
    specs = []
    for name, s in cfg["faults"].items():
        if s.get("enabled", True):
            specs.append((name, s.get("severity_levels", [1.0]), s["label"]))
    for comp in cfg.get("compound_faults", []) or []:
        if comp.get("enabled", True):
            specs.append((comp["name"], comp.get("severity_levels", [0.5]), comp["label"]))

    rng = np.random.default_rng(args.seed)
    run_id = 0
    rows = []
    for fault, sev_levels, label in specs:
        for r in range(args.runs_per_class):
            sev = float(rng.choice(sev_levels))
            Hs, Tp = SEA_STATES[rng.integers(len(SEA_STATES))]
            run_id += 1
            df = synth_run(fault, sev, Hs, Tp, seed=args.seed + run_id)
            fn = DATA_RAW / (f"run_{run_id:05d}__{fault}_sev{round(sev*100):02d}"
                             f"_Hs{round(Hs*100):03d}_Tp{round(Tp*10):03d}.csv")
            df.to_csv(fn, index=False)
            rows.append({"runID": run_id, "fault": fault, "label": label,
                         "severity": sev, "Hs": Hs, "Tp": Tp, "file": str(fn)})
    pd.DataFrame(rows).to_csv(DATA_RAW / "manifest.csv", index=False)
    log.info("wrote %d synthetic runs across %d classes to %s",
             run_id, len(specs), DATA_RAW)


if __name__ == "__main__":
    main()
