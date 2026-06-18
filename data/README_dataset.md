# RM3 Multi-Fault WEC Dataset — Data Card

> **Status: TEMPLATE.** Fill the bracketed fields once the real WEC-Sim runs are generated.
> This card travels with the dataset on release (Zenodo / IEEE DataPort) and is the headline
> contribution of the thesis (build guide §5.6, §8).

## Overview
- **Device:** RM3 (Reference Model 3) two-body floating point absorber, simulated in WEC-Sim.
- **Purpose:** supervised + unsupervised fault detection / classification for WEC condition monitoring.
- **Novelty:** a *unified* multi-fault set spanning PTO **and** mooring subsystems, including
  **compound** faults — no public dataset covers multiple WEC subsystems together.

## Generation
- **Simulator:** WEC-Sim (MATLAB + Simulink + Simscape Multibody), RM3 example. Version: [x.x].
- **Waves:** irregular, JONSWAP. Sea-state matrix from NDBC buoys [stations / years].
- **Sim settings:** dt = 0.1 s, ramp = 100 s, end = 400 s [confirm]. Initial transient dropped in windowing.
- **Mooring model:** linear stiffness matrix (baseline) / MoorDyn [if upgraded].

## Channels (per run CSV)
| column | unit | description |
|---|---|---|
| `t` | s | time |
| `z1` | m | float heave |
| `z2` | m | spar heave |
| `rel` | m | relative heave (z1 − z2), drives the PTO |
| `vrel` | m/s | relative velocity |
| `Fpto` | N | PTO force (heave) |
| `Ppto` | W | PTO instantaneous power |
| `Tmoor` | N | mooring line tension (heave component) |

## Fault classes
| label | name | subsystem | severities | description |
|---|---|---|---|---|
| 0 | healthy | — | 1.0 | nominal RM3 |
| 1 | pto_damping_loss | PTO | 0.8, 0.6, 0.4, 0.2 | PTO damping reduced (seal wear / fluid loss) |
| 2 | mooring_stiffness_loss | mooring | 0.75, 0.5, 0.25 | mooring stiffness reduced (corrosion / partial failure) |
| 4 | pto_plus_mooring | PTO + mooring | 0.5 | compound fault (both degraded) |
| 3 | generator_fault | generator | — | **reserved / future work** (not in this release) |

Severity = multiplicative factor on the nominal parameter (1.0 = healthy).

## Variants
- **Noise variants:** Gaussian sensor noise at relative-std levels {0, 0.01, 0.02, 0.05, 0.1, 0.2}.
- **Degradation-trajectory runs** (for early detection): severity ramps within a single run.

## Files
- `run_NNNNN__<fault>_sevSS_HsHHH_TpTTT.csv` — one simulation run; metadata encoded in the name.
- `manifest.csv` — index of all runs with labels, severities, sea states.

## How to load
```python
from src.windowing import build_dataset
X, y, runid = build_dataset(win=512, stride=256)   # leakage-free windows + group ids
```

## Provenance / honesty
- **Simulation only.** Models trained here are not validated on real WEC hardware — sim-to-real
  transfer is explicitly future work.
- License: **CC BY 4.0** [confirm].
- DOI: [assigned on Zenodo/IEEE DataPort deposit].
