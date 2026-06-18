# Direction A — Simulated Fault Detection & Condition Monitoring of a Wave Energy Converter

**Author:** V Vissal · MTech, Data & Computational Science, IIT Jodhpur
**Thesis direction (professor-approved):** Simulated fault detection and condition monitoring of a WEC
**Companion file:** `DIRECTION_B_AR_vs_DL_BUILD_GUIDE.md` (the second approved direction)
**This file is a build guide** to be used with an LLM coding assistant (e.g. Claude) inside VS Code. It combines (a) a precise checklist of what to do, and (b) starter code you can run and extend.

---

## 0. READ THIS FIRST — instructions to the LLM assistant

You are assisting a developer to build this thesis project end to end. Assume and apply **professional-level knowledge** of:

- **Machine learning & deep learning:** 1D-CNN, LSTM/GRU, temporal convolutional networks, autoencoders (for anomaly detection), one-class SVM / isolation forest, class-imbalance handling (class weights, focal loss, SMOTE for tabular features), hyperparameter tuning (Optuna), cross-validation that respects time/grouping, calibration, SHAP explainability.
- **Time-series & signal processing:** windowing/segmentation, resampling, filtering (Butterworth, band-pass), spectral features (FFT, PSD, spectrogram), statistical features (`tsfresh`), normalisation/standardisation done **without leakage** (fit scaler on train only).
- **MATLAB / Simulink / WEC-Sim:** how WEC-Sim couples MATLAB + Simulink + Simscape Multibody; the RM3 two-body point-absorber model; `wecSimInputFile.m` structure (`waves`, `body`, `pto`, `mooring`, `simu` objects); Multiple Condition Runs (MCR); MoorDyn coupling for moorings; PTO-Sim for the power-take-off chain; how to script batch runs and export time-series to `.mat`/`.csv`.
- **Software engineering:** clean repo structure, reproducible seeds, config files, saving/loading models, logging, unit-testing data shapes.

**Hard project constraints (do not violate):**
- Software-only. No hardware, no wave tank, no sea trials.
- Free public data only (NDBC, CDIP) or self-generated simulation data.
- **MATLAB is "MATLAB Online" (browser-based).** Prefer workflows that run there. Only suggest a local install if something genuinely cannot run on MATLAB Online — and flag it explicitly.
- CPU-only for the ML side. Favour light models (1D-CNN, LSTM, classical ML). Keep anything heavy optional.
- One semester, ~2–4 hrs/day. Keep scope finishable; respect the staged plan and go/no-go gates in §9.

**Fixed technical decisions (do not change without asking the developer):**
- **WEC model: RM3 (Reference Model 3)** — the two-body floating point absorber (a float oscillating around a spar/plate), the standard WEC-Sim tutorial device. Everything is built around RM3.
- **Core faults for this thesis: (1) PTO fault and (2) Mooring fault.** Architect the code so a **third fault (generator/electrical)** and **compound faults** can be added later without redesign (see §5.4 — the fault definitions must be data-driven config entries, not hard-coded branches).

---

## 1. What this project is (plain statement)

A wave energy converter is a machine; machines develop faults. Caught early they are cheap to fix; caught late they cause expensive downtime. There is **no public dataset of faulty-WEC signals**, so this project:

1. **Generates** healthy and faulty RM3 signals in WEC-Sim, driven by realistic sea states from NDBC/CDIP buoy data.
2. **Trains** time-series models (1D-CNN, LSTM) to (a) detect that a fault exists and (b) classify which fault it is.
3. **Adds the innovative angles** that lift this above existing single-subsystem fault papers (see §2).
4. **Releases** the labelled dataset as a contribution (Zenodo/IEEE DataPort with a DOI).

---

## 2. Innovative angles (what makes this impactful, not incremental)

Existing work each covers **one** subsystem (Tang 2020 components; Subramanian & Zou 2024 mooring; Kang 2024 gearbox). The thesis differentiates on these verified-open angles:

1. **Unified multi-fault dataset (PTO + mooring together, incl. compound faults).** No public dataset spans multiple subsystems together — this is the headline contribution. (Highest impact, novelty-confidence HIGH.)
2. **Early / incipient detection with lead-time quantification.** Simulate gradual degradation and measure *how many minutes/cycles before functional failure* the fault becomes detectable. Essentially unstudied for WECs. (HIGH.)
3. **Unsupervised anomaly detection for unseen faults.** Train an autoencoder/one-class model on healthy data only; flag fault types never seen in training. Distinguishes from crowded supervised-only work. (MEDIUM-HIGH.)
4. **Noise-robustness as a first-class result**, not an afterthought — escalating sensor noise / dropout and report degradation curves. (MEDIUM.)
5. **SHAP explainability tied to failure physics** — e.g. mooring fault → low-frequency surge drift; PTO fault → power/velocity phase change. (MEDIUM, strong as secondary contribution.)

> Honesty note for the thesis: the ML methods themselves (1D-CNN, LSTM, autoencoders) are standard. The novelty is the **unified multi-fault dataset + early-detection lead-time + unseen-fault detection on a WEC**, not a new algorithm. State it that way.

---

## 3. Key literature (anchors + DOIs)

| Paper | Year | Venue / DOI | Role |
|---|---|---|---|
| Mortazavizadeh, Yazdanpanah, Campos Gaona, Anaya-Lara — "Fault Diagnosis and Condition Monitoring in WECs: A Review" | 2023 | *Energies* 16(19):6777 · DOI **10.3390/en16196777** (verified) | Motivation: "57% of OpEx is maintenance"; wind CM detects faults "months in advance", 15–20% savings |
| Tang, Huang, Lindbeck, Lizza et al. — "WEC fault modelling and condition monitoring: a graph-theoretic approach" | 2020 | *IET Electric Power Applications* 14(5) · DOI **confirm on publisher page** | First faulty-component models in WEC-Sim; no public deposit found |
| Subramanian, Zou, Zhou, Su — "Data-Driven Fault Diagnosis of Mooring Systems in WECs" | 2024 | *OCEANS 2024* · DOI **confirm on publisher page** | Mooring faults via MoorDyn+WEC-Sim on RM3; closest prior art for the mooring half |
| Kang, Zhu, Shen, Li — "Fault diagnosis of a WEC gearbox … CNN-LSTM" | 2024 | *Renewable Energy* 231:121022 · DOI **10.1016/j.renene.2024.121022** (verify) | CNN-LSTM on simulated gearbox vibration; notes it did NOT cover compound/severity |
| Said, Papini/Sardá, Faedo, Ringwood — "Fault diagnosis and fault-tolerant control in wave energy: a perspective" | 2024 | *Renewable & Sustainable Energy Reviews* 199 · DOI **confirm on publisher page** | Framework/perspective; situates diagnosis→prognosis→FTC |

> Always click each "confirm on publisher page" DOI before citing in the thesis. A wrong DOI is worse than none.

Useful public building blocks (none is a multi-fault labelled set — that's the gap): MHKDR LUPA WEC-Sim+MoorDyn model (DOI 10.15473/2481242, healthy only); SWELL experimental array data (Mendeley, no fault labels); WEC-Sim itself (OSTI 1887272).

---

## 4. Repository structure

```
wec-fault-detection/
├── README.md
├── requirements.txt
├── config/
│   ├── sea_states.yaml          # Hs/Tp matrix derived from buoy data
│   └── faults.yaml              # fault definitions (DATA-DRIVEN — see §5.4)
├── matlab/                      # runs in MATLAB Online
│   ├── run_batch.m              # top-level batch driver (MCR-style)
│   ├── wecSimInputFile_template.m
│   ├── inject_fault.m           # applies a fault config to the input objects
│   └── export_timeseries.m      # writes per-run CSV to data/raw/
├── data/
│   ├── raw/                     # CSVs exported from WEC-Sim (git-ignored)
│   ├── processed/               # windowed arrays (.npz)
│   └── README_dataset.md        # the data card for release
├── src/
│   ├── ingest.py                # load raw CSV, basic cleaning
│   ├── sea_states.py            # build sea-state matrix from NDBC/CDIP
│   ├── windowing.py             # segment into fixed windows (no leakage)
│   ├── features.py              # optional tsfresh/spectral features
│   ├── datasets.py              # torch Dataset/DataLoader, splits
│   ├── models.py                # 1D-CNN, LSTM, autoencoder
│   ├── train.py                 # training loop, class weights, early stop
│   ├── evaluate.py              # metrics, confusion matrix, noise sweep
│   ├── anomaly.py               # autoencoder/one-class unseen-fault detection
│   ├── leadtime.py              # incipient-detection lead-time experiment
│   ├── explain.py               # SHAP + physics mapping
│   └── utils.py                 # seeds, logging, config loading
├── notebooks/                   # exploration only; real code lives in src/
└── outputs/                     # figures, tables, trained models
```

---

## 5. Part 1 — Generating the dataset (MATLAB Online + WEC-Sim)

### 5.1 Checklist
- [ ] **Verify the toolchain on MATLAB Online first (go/no-go #1).** Confirm Simulink + Simscape Multibody are available, install/clone WEC-Sim, and run the **stock RM3 regular-wave example unchanged**. If RM3 will not run on MATLAB Online, stop and tell the developer — this is the one true blocker, and only then consider a local install.
- [ ] Get WEC-Sim (github.com/WEC-Sim/WEC-Sim) and the RM3 example into the MATLAB Online workspace.
- [ ] Build the sea-state matrix from buoy data (§5.3).
- [ ] Implement data-driven fault injection (§5.4).
- [ ] Batch-run healthy + faulty + compound cases; export per-run CSV (§5.5).
- [ ] Assemble + label + write the data card (§5.6).

### 5.2 RM3 quick facts (so the model is fixed and understood)
RM3 is a **two-body point absorber**: a surface **float** reacts against a submerged **spar/heave-plate**. Power is taken from the **relative heave motion** between the two bodies via the PTO. It is moored to the seabed. Signals you will log per run: float & spar heave (and pitch/surge if desired), **relative displacement & velocity**, **PTO force & instantaneous power**, and **mooring line tension(s)**. These are the channels the ML models consume.

### 5.3 Sea states from buoy data (`src/sea_states.py` → `config/sea_states.yaml`)
Use real Hs/Tp ranges so simulated signals resemble reality. A defensible matrix (mirrors NREL practice): **Hs 0.25–3.75 m**, **Tp 5–13 s**, irregular waves (JONSWAP). Derive the actual ranges from a couple of NDBC buoys.

```python
# src/sea_states.py  — derive an Hs/Tp matrix from NDBC standard meteorological data
import io, requests, numpy as np, pandas as pd, yaml

def fetch_ndbc_year(station: str, year: int) -> pd.DataFrame:
    """NDBC historical standard met file (yearly). Returns df with WVHT (Hs), DPD/APD (period)."""
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={station}h{year}.txt.gz&dir=data/historical/stdmet/"
    raw = requests.get(url, timeout=60).text
    # first row = headers, second row = units; 'MM'/99/999 are missing codes
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", skiprows=[1], na_values=["MM", 99, 999, 9999])
    return df

def build_sea_state_matrix(stations, years, n_hs=8, n_tp=5):
    frames = []
    for s in stations:
        for y in years:
            try: frames.append(fetch_ndbc_year(s, y)[["WVHT", "DPD"]])
            except Exception as e: print(f"skip {s}{y}: {e}")
    d = pd.concat(frames).dropna()
    d = d[(d.WVHT > 0) & (d.WVHT < 12) & (d.DPD > 2) & (d.DPD < 25)]
    hs = np.round(np.linspace(max(0.25, d.WVHT.quantile(.05)), min(3.75, d.WVHT.quantile(.95)), n_hs), 2)
    tp = np.round(np.linspace(d.DPD.quantile(.10), d.DPD.quantile(.90), n_tp), 1)
    grid = [{"Hs": float(h), "Tp": float(t)} for h in hs for t in tp]
    yaml.safe_dump({"sea_states": grid}, open("config/sea_states.yaml", "w"))
    print(f"{len(grid)} sea states written")
    return grid

if __name__ == "__main__":
    build_sea_state_matrix(["46022", "46050"], [2018, 2019])  # example Pacific buoys
```

### 5.4 Data-driven fault definitions (`config/faults.yaml`) — the key to future-extensibility
Faults are **config entries**, never hard-coded `if` branches. Adding a generator fault later = adding a YAML block, no code rewrite.

```yaml
# config/faults.yaml
faults:
  healthy:
    label: 0
    description: "Nominal RM3, no degradation"
    params: {}

  pto_damping_loss:                 # FAULT 1 (core)
    label: 1
    subsystem: pto
    description: "PTO damping coefficient reduced (seal wear / fluid loss)"
    # multiply nominal PTO linear damping by these factors (1.0 = healthy)
    severity_levels: [0.8, 0.6, 0.4, 0.2]
    target: "pto(1).c"             # PTO damping coefficient in the input file

  mooring_stiffness_loss:           # FAULT 2 (core)
    label: 2
    subsystem: mooring
    description: "Mooring line stiffness reduced / line degraded (corrosion, partial failure)"
    severity_levels: [0.75, 0.5, 0.25]
    target: "mooring_stiffness_scale"

  # ---- FUTURE (architecture already supports; do not build yet) ----
  generator_fault:
    label: 3
    subsystem: generator
    description: "Generator electrical fault / torque oscillation (via PTO-Sim)"
    enabled: false

compound_faults:                    # the novel part: simultaneous faults
  - name: pto_plus_mooring
    label: 4
    combine: [pto_damping_loss, mooring_stiffness_loss]
    enabled: true
```

```matlab
% matlab/inject_fault.m — apply one fault config to the WEC-Sim input objects
function inject_fault(faultName, severity)
% Called inside wecSimInputFile after the healthy objects are defined.
% Reads config/faults.yaml semantics but implemented explicitly for the two core faults.
    global pto body mooring
    switch faultName
        case 'healthy'
            % no change
        case 'pto_damping_loss'
            pto(1).c = pto(1).c * severity;          % reduce PTO linear damping
        case 'mooring_stiffness_loss'
            % if using a linear mooring matrix, scale its stiffness;
            % if using MoorDyn, scale line stiffness in the MoorDyn input file instead.
            mooring(1).matrix(3,3) = mooring(1).matrix(3,3) * severity;  % heave stiffness example
        case 'pto_plus_mooring'
            pto(1).c = pto(1).c * severity;
            mooring(1).matrix(3,3) = mooring(1).matrix(3,3) * severity;
        otherwise
            error('Unknown fault: %s', faultName);
    end
end
```

> **Mooring modelling choice:** start with WEC-Sim's **linear mooring matrix** (simplest, runs fast on MATLAB Online) and scale its stiffness for the mooring fault. **MoorDyn** gives higher-fidelity line dynamics (matching Subramanian & Zou) but is heavier — make it a documented upgrade path, not the first build.

### 5.5 Batch driver + export (`matlab/run_batch.m`, `export_timeseries.m`)
```matlab
% matlab/run_batch.m — generate the whole dataset
faults = {'healthy','pto_damping_loss','mooring_stiffness_loss','pto_plus_mooring'};
severity.pto_damping_loss        = [0.8 0.6 0.4 0.2];
severity.mooring_stiffness_loss  = [0.75 0.5 0.25];
severity.pto_plus_mooring        = [0.5];        % representative compound case
severity.healthy                 = [1.0];

seaStates = yaml_read('config/sea_states.yaml');  % or hardcode a small grid for first runs
runID = 0;
for f = 1:numel(faults)
  fname = faults{f};
  for sev = severity.(fname)
    for ss = 1:numel(seaStates)
      runID = runID + 1;
      Hs = seaStates(ss).Hs;  Tp = seaStates(ss).Tp;
      % wecSimInputFile_template.m reads Hs,Tp,fname,sev from the workspace,
      % defines RM3 objects, then calls inject_fault(fname,sev) before wecSim.
      run('wecSimInputFile_template.m');          % sets up objects + fault
      wecSim;                                     % run the simulation
      export_timeseries(output, fname, faults_label(fname), sev, Hs, Tp, runID);
    end
  end
end
```
```matlab
% matlab/export_timeseries.m — one tidy CSV per run into data/raw/
function export_timeseries(output, faultName, label, severity, Hs, Tp, runID)
    t   = output.bodies(1).time;
    z1  = output.bodies(1).position(:,3);          % float heave
    z2  = output.bodies(2).position(:,3);          % spar heave
    rel = z1 - z2;                                 % relative heave (drives PTO)
    vrel= [0; diff(rel)] ./ [1; diff(t)];          % relative velocity
    Fpto= output.ptos(1).forceTotal(:,3);          % PTO force (heave)
    Ppto= output.ptos(1).power(:,3);               % PTO instantaneous power
    Tmoor = output.mooring.forceTotal(:,3);        % mooring tension (heave comp.)
    T = table(t, z1, z2, rel, vrel, Fpto, Ppto, Tmoor);
    meta = sprintf('fault=%s,label=%d,sev=%.2f,Hs=%.2f,Tp=%.2f', faultName,label,severity,Hs,Tp);
    fn = sprintf('data/raw/run_%05d__%s_sev%02d_Hs%03d_Tp%03d.csv', ...
                 runID, faultName, round(severity*100), round(Hs*100), round(Tp*10));
    writetable(T, fn);
    % also append meta to data/raw/manifest.csv
end
```

### 5.6 Dataset assembly + data card (release contribution)
- [ ] Concatenate `data/raw/manifest.csv` listing every run with its labels.
- [ ] Write `data/README_dataset.md`: device (RM3), channels, sampling rate, sea-state matrix, fault definitions + severities, noise variants, license (CC BY 4.0), and how to load.
- [ ] Deposit on **Zenodo** or **IEEE DataPort** → get a DOI. Cite it in the thesis as a first-class contribution.

---

## 6. Part 2 — The ML pipeline (Python, CPU-friendly)

### 6.1 `requirements.txt`
```
numpy
pandas
scipy
scikit-learn
torch            # CPU build is fine
matplotlib
pyyaml
tsfresh          # optional engineered features
shap
optuna           # optional tuning
```

### 6.2 Windowing without leakage (`src/windowing.py`)
```python
import numpy as np, pandas as pd, glob, os, re

CHANNELS = ["z1", "z2", "rel", "vrel", "Fpto", "Ppto", "Tmoor"]

def load_run(path):
    df = pd.read_csv(path)
    m = re.search(r"__(?P<fault>[a-z_]+)_sev(?P<sev>\d+)_Hs(?P<hs>\d+)_Tp(?P<tp>\d+)", os.path.basename(path))
    meta = m.groupdict() if m else {}
    return df, meta

def window_run(df, win=512, stride=256):
    """Slice one run into [win, n_channels] windows. Drop the initial transient."""
    x = df[CHANNELS].to_numpy(dtype=np.float32)
    x = x[2000:]                       # discard ramp-up transient (tune to your dt)
    out = []
    for s in range(0, len(x) - win, stride):
        out.append(x[s:s+win])
    return np.stack(out) if out else np.empty((0, win, len(CHANNELS)), np.float32)

def build_dataset(raw_glob="data/raw/run_*.csv", win=512, stride=256, label_map=None):
    X, y, runid = [], [], []
    for i, path in enumerate(sorted(glob.glob(raw_glob))):
        df, meta = load_run(path)
        w = window_run(df, win, stride)
        if len(w) == 0: continue
        lbl = label_map[meta["fault"]]
        X.append(w); y.append(np.full(len(w), lbl)); runid.append(np.full(len(w), i))
    X = np.concatenate(X); y = np.concatenate(y); runid = np.concatenate(runid)
    np.savez_compressed("data/processed/dataset.npz", X=X, y=y, runid=runid)
    return X, y, runid
```
> **Leakage rule (critical):** split **by run**, not by window. All windows from one simulation must fall entirely in train OR test, never both. Fit the scaler on the training windows only.

### 6.3 Splits + scaling (`src/datasets.py`)
```python
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

class WindowDS(Dataset):
    def __init__(self, X, y): self.X = torch.tensor(X).transpose(1,2); self.y = torch.tensor(y).long()
    # X shape -> [N, channels, time] for 1D-CNN
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def grouped_split(X, y, runid, test_size=0.25, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups=runid))
    mu, sd = X[tr].mean((0,1), keepdims=True), X[tr].std((0,1), keepdims=True) + 1e-8
    Xtr, Xte = (X[tr]-mu)/sd, (X[te]-mu)/sd        # scaler fit on train only
    return Xtr, y[tr], Xte, y[te], (mu, sd)
```

### 6.4 Models (`src/models.py`) — light, CPU-friendly
```python
import torch, torch.nn as nn

class CNN1D(nn.Module):
    """Compact 1D-CNN for fault detection/classification. Good at local signal motifs."""
    def __init__(self, n_ch, n_cls, k=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, k, padding=k//2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, k, padding=k//2),   nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, k, padding=k//2),   nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))
    def forward(self, x): return self.net(x)

class LSTMClf(nn.Module):
    """LSTM for temporal evolution. Expects [N, channels, time] -> transpose inside."""
    def __init__(self, n_ch, n_cls, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))
    def forward(self, x):
        x = x.transpose(1, 2)              # [N, time, channels]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

class AE(nn.Module):
    """1D conv autoencoder for UNSUPERVISED anomaly detection (train on healthy only)."""
    def __init__(self, n_ch):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(n_ch,32,7,2,3), nn.ReLU(), nn.Conv1d(32,16,7,2,3), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose1d(16,32,7,2,3,1), nn.ReLU(),
                                 nn.ConvTranspose1d(32,n_ch,7,2,3,1))
    def forward(self, x): return self.dec(self.enc(x))
```

### 6.5 Training (`src/train.py`) — with class weights for imbalance
```python
import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

def train(model, ds_tr, ds_val, n_cls, y_tr, epochs=40, lr=1e-3, bs=64, device="cpu"):
    cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_tr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = DataLoader(ds_val, batch_size=bs)
    best, patience, wait = 1e9, 6, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad(); loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward(); opt.step()
        # validation
        model.eval(); vloss = 0
        with torch.no_grad():
            for xb, yb in dl_va: vloss += crit(model(xb.to(device)), yb.to(device)).item()
        vloss /= len(dl_va)
        print(f"epoch {ep}  val_loss {vloss:.4f}")
        if vloss < best: best, wait = vloss, 0; torch.save(model.state_dict(), "outputs/best.pt")
        else: wait += 1
        if wait >= patience: print("early stop"); break
    model.load_state_dict(torch.load("outputs/best.pt")); return model
```

### 6.6 Evaluation + noise robustness (`src/evaluate.py`)
```python
import numpy as np, torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, X, y, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X).transpose(1,2).to(device))
        pred = logits.argmax(1).cpu().numpy()
    print(classification_report(y, pred, digits=3))
    return confusion_matrix(y, pred)

def noise_sweep(model, X, y, levels=(0.0,0.01,0.02,0.05,0.1,0.2), device="cpu"):
    """Add Gaussian noise of increasing std (relative to channel std); report accuracy curve."""
    rng = np.random.default_rng(0); out = {}
    chan_std = X.std((0,1), keepdims=True)
    for lv in levels:
        Xn = X + rng.normal(0, lv, X.shape).astype(np.float32) * chan_std
        with torch.no_grad():
            pred = model(torch.tensor(Xn).transpose(1,2).to(device)).argmax(1).cpu().numpy()
        out[lv] = float((pred == y).mean())
    return out   # plot this as the robustness curve (headline result, not afterthought)
```

### 6.7 Unsupervised unseen-fault detection (`src/anomaly.py`)
```python
# Train AE on HEALTHY windows only; a held-out fault type the model never saw should
# produce HIGH reconstruction error -> detected as anomaly without ever being labelled.
import torch, numpy as np
def recon_error(ae, X, device="cpu"):
    ae.eval()
    with torch.no_grad():
        x = torch.tensor(X).transpose(1,2).to(device)
        e = ((ae(x) - x)**2).mean(dim=(1,2)).cpu().numpy()
    return e
# threshold = healthy_train_error.mean() + 3*std ; report ROC-AUC of healthy vs unseen-fault.
```

### 6.8 Early / incipient detection (`src/leadtime.py`)
- [ ] Generate **degradation-trajectory** runs: severity worsens over time within a single run (e.g. PTO damping factor ramps 1.0 → 0.2).
- [ ] Slide the classifier along the run; record the time index where it first flags the fault confidently.
- [ ] Define **lead time** = (time of functional failure) − (time of first confident detection). Report lead time vs sea state and vs degradation rate. This quantified curve is a novel deliverable.

### 6.9 Explainability (`src/explain.py`)
```python
import shap, torch, numpy as np
def shap_summary(model, X_background, X_explain):
    # GradientExplainer works for torch models; reduce background size for CPU speed
    e = shap.GradientExplainer(model, torch.tensor(X_background).transpose(1,2))
    sv = e.shap_values(torch.tensor(X_explain[:200]).transpose(1,2))
    return sv   # aggregate |SHAP| per channel -> which signal reveals each fault
# Then MAP to physics: mooring fault should weight Tmoor + low-freq surge; PTO fault -> Ppto/vrel.
```

---

## 7. Experimental protocol (write this into the thesis methods)
- Split **by run**, stratified by fault class; fixed seeds; scaler fit on train only.
- Report per-class precision/recall/F1 + confusion matrix for **two tasks**: binary (fault/no-fault) and multi-class (which fault).
- Robustness: accuracy-vs-noise curve; accuracy on **unseen sea states**.
- Unsupervised: ROC-AUC for detecting a **held-out fault type** never in training.
- Early detection: lead-time distribution across runs.
- Explainability: per-fault channel attributions vs expected physics.

---

## 8. Final contributions to claim
1. **Open unified multi-fault RM3 dataset** (PTO + mooring + compound), with DOI — the lasting contribution.
2. **Early-detection lead-time benchmark** for WEC faults.
3. **Unsupervised detection of unseen faults.**
4. **Noise-robustness study** as a first-class result.
5. **SHAP-to-physics mapping.**
6. Thesis + a conference paper (OCEANS / EWTEC).

---

## 9. One-semester timeline + go/no-go gates
- **Weeks 1–2 — Toolchain (GATE 1):** RM3 stock example runs on MATLAB Online; Python repo skeleton builds. *No-go → consider local MATLAB install; tell developer.*
- **Weeks 3–5 — Dataset core:** healthy + PTO + mooring single-fault runs across the sea-state matrix; export + manifest. **(GATE 2: dataset sane, classes separable in a quick baseline.)**
- **Weeks 6–7 — Compound faults + noise variants; baseline 1D-CNN & LSTM classification.**
- **Weeks 8–9 — Early-detection lead-time + unsupervised unseen-fault experiments.**
- **Weeks 10–11 — Robustness sweeps + SHAP/physics. (GATE 3: at least one clear headline result.)**
- **Weeks 12–14 — Dataset release (Zenodo DOI), thesis write-up, paper draft.**
- **Weeks 15–16 — Buffer / reproducibility check.**

**Pivots:** if WEC-Sim setup exceeds ~2 weeks on MATLAB Online → try the local install once; if still stuck → fall back to the pure-Python WecOptTool/Capytaine path (lower fidelity, document the trade-off). If compound faults are unstable → ship PTO + mooring single-faults as the core and mark compound as future work.

---

## 10. Honesty / scope reminders
- The ML methods are standard; the novelty is the **unified multi-fault WEC dataset + early-detection + unseen-fault detection**. Frame claims that way.
- Models trained on simulation may not transfer to real WECs — do **not** claim real-world readiness; note sim-to-real as future work.
- Before claiming "first unified multi-fault dataset," do a final repository sweep (Zenodo, IEEE DataPort, Mendeley) to confirm none was published meanwhile.
- Confirm every "confirm on publisher page" DOI before the thesis goes in.
