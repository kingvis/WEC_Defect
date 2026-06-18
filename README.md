# WEC Fault Detection — Simulated Fault Detection & Condition Monitoring of an RM3 Wave Energy Converter

**Author:** V Vissal · MTech, Data & Computational Science, IIT Jodhpur
**Thesis direction (professor-approved):** Simulated fault detection and condition monitoring of a WEC.

This repo implements **Direction A** from `DIRECTION_A_fault_detection_BUILD_GUIDE.md`. It has two halves:

1. **Data generation (MATLAB Online + WEC-Sim):** generate healthy and faulty RM3 signals across realistic
   sea states. Code lives in [matlab/](matlab/).
2. **ML pipeline (Python, CPU-friendly):** window the signals, train 1D-CNN / LSTM classifiers and an
   autoencoder, and run the early-detection, unseen-fault, noise-robustness and SHAP experiments.
   Code lives in [src/](src/).

## Quick start (Python side — works *now*, no MATLAB needed)

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# bash:                source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic RM3-like data so the whole pipeline is testable before WEC-Sim is wired up
python -m tools.make_synthetic_data --runs-per-class 6

# 2. Run the end-to-end pipeline (window -> split -> train CNN -> evaluate -> noise sweep)
python -m tools.run_pipeline --model cnn --epochs 8
```

The synthetic generator stands in for WEC-Sim output. Once real CSVs from WEC-Sim land in `data/raw/`
with the same schema, the exact same pipeline runs on real data — no code changes.

## Data schema (one CSV per simulation run)

Columns: `t, z1, z2, rel, vrel, Fpto, Ppto, Tmoor` (see [src/windowing.py](src/windowing.py) `CHANNELS`).
Filenames encode metadata: `run_00042__pto_damping_loss_sev60_Hs150_Tp080.csv`.

| channel | meaning |
|---|---|
| `t`    | time (s) |
| `z1`   | float heave (m) |
| `z2`   | spar heave (m) |
| `rel`  | relative heave z1 − z2 (drives the PTO) |
| `vrel` | relative velocity |
| `Fpto` | PTO force (heave) |
| `Ppto` | PTO instantaneous power |
| `Tmoor`| mooring line tension (heave component) |

## Faults (data-driven, see [config/faults.yaml](config/faults.yaml))

| label | fault | subsystem |
|---|---|---|
| 0 | healthy | — |
| 1 | pto_damping_loss | PTO |
| 2 | mooring_stiffness_loss | mooring |
| 4 | pto_plus_mooring (compound) | PTO + mooring |

Label 3 (generator) is reserved for future work — architecture supports it without a rewrite.

## Repo layout

See §4 of the build guide. Key directories: `config/`, `matlab/`, `src/`, `tools/`, `data/`, `outputs/`.

## Status / gates

See §9 of the build guide. **Gate 1** (RM3 stock example runs on MATLAB Online) is the critical blocker —
verification checklist in [matlab/GATE1_CHECKLIST.md](matlab/GATE1_CHECKLIST.md).
