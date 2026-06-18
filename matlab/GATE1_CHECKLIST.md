# Gate 1 — MATLAB Online + WEC-Sim toolchain verification

**This is the one true blocker (build guide §5.1, §9).** Do not invest in fault injection or batch
runs until the *stock* RM3 example runs unchanged on MATLAB Online. Work through this top to bottom.

## Step 0 — Confirm toolboxes (MATLAB Online)
- [ ] Open MATLAB Online (matlab.mathworks.com) and run `ver` in the Command Window.
- [ ] Confirm **Simulink** is listed.
- [ ] Confirm **Simscape** and **Simscape Multibody** are listed.
- [ ] If Simscape Multibody is missing, WEC-Sim cannot run → record this and tell the developer
      (this is the no-go that justifies considering a local install).

## Step 1 — Get WEC-Sim into the workspace
- [ ] Clone or download WEC-Sim: `https://github.com/WEC-Sim/WEC-Sim`
- [ ] In MATLAB: `cd` into the WEC-Sim source folder and run `addWecSimSource` (adds it to the path).
- [ ] Run `wecSimVersion` (or check `source/` is on the path) to confirm it resolved.

## Step 2 — Run the STOCK RM3 example UNCHANGED
- [ ] `cd` to `WEC-Sim/examples/RM3`
- [ ] Run `wecSim`
- [ ] Confirm it completes without error and produces an `output` object.
- [ ] Spot-check: `output.bodies(1).position` is non-empty; plots look like oscillating heave.

✅ **If this passes, Gate 1 is cleared.** Proceed to wire in this repo's files.

## Step 3 — Wire in this repo's fault-injection layer
- [ ] Copy `RM3.slx`, `hydroData/`, and `geometry/` from the RM3 example into this repo's `matlab/`
      folder (or add their paths), so `wecSimInputFile_template.m` can find them.
- [ ] Put `matlab/` on the MATLAB path: `addpath(genpath('matlab'))` from the repo root.
- [ ] Smoke test a single faulted run:
  ```matlab
  Hs = 2.0; Tp = 8.0; faultName = 'pto_damping_loss'; sev = 0.4;
  run('matlab/wecSimInputFile_template.m');   % should print the fault line
  wecSim;
  export_timeseries(output, faultName, 1, sev, Hs, Tp, 1);
  ```
- [ ] Confirm a CSV appears in `data/raw/` with the 8 expected columns.

## Step 4 — Full batch
- [ ] Run `run_batch.m` from the repo root.
- [ ] Confirm `data/raw/manifest.csv` lists every run and CSVs cover all 4 fault classes.

## Notes / gotchas
- **`pto(1).c`**: the template uses `.c` as the damping alias `inject_fault` scales, then copies it back
  into `pto(1).damping` (the field WEC-Sim actually reads). Verify the field name matches your WEC-Sim
  version — newer versions may use `pto(1).damping` directly.
- **Mooring force logging**: a linear mooring matrix may not populate `output.mooring.forceTotal`.
  `export_timeseries.m` guards this with zeros; if `Tmoor` is all-zero, switch to MoorDyn (upgrade path)
  or derive tension from mooring displacement × stiffness.
- **Runtime**: `endTime=400, dt=0.1` keeps each run light. If MATLAB Online times out on long batches,
  reduce `endTime`, shrink the sea-state grid, or run faults in separate sessions.
- **If WEC-Sim setup exceeds ~2 weeks** (guide §9 pivot): try a local install once; if still stuck,
  fall back to the pure-Python WecOptTool / Capytaine path (lower fidelity — document the trade-off).
