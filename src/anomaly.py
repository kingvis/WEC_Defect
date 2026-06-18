"""Unsupervised unseen-fault detection with a conv autoencoder.

Train the AE on HEALTHY windows only. A fault type the model never saw should produce HIGH
reconstruction error -> detected as an anomaly without ever being labelled (build guide §6.7).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .models import AE
from .utils import get_logger

log = get_logger("anomaly")


def _to_tensor(X):
    return torch.tensor(np.asarray(X), dtype=torch.float32).transpose(1, 2)


def train_ae(X_healthy, n_ch, epochs=30, lr=1e-3, bs=64, device="cpu"):
    """Train the autoencoder on healthy windows only."""
    ae = AE(n_ch).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    crit = nn.MSELoss()
    dl = DataLoader(TensorDataset(_to_tensor(X_healthy)), batch_size=bs, shuffle=True)
    for ep in range(epochs):
        ae.train()
        tot = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            out = ae(xb)
            # ConvTranspose may differ by a sample; align lengths defensively.
            n = min(out.shape[-1], xb.shape[-1])
            loss = crit(out[..., :n], xb[..., :n])
            loss.backward()
            opt.step()
            tot += loss.item()
        log.info("AE epoch %d  loss %.5f", ep, tot / max(1, len(dl)))
    return ae


def recon_error(ae, X, device="cpu", bs=256) -> np.ndarray:
    """Per-window mean-squared reconstruction error."""
    ae.eval()
    Xt = _to_tensor(X)
    errs = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            xb = Xt[i:i + bs].to(device)
            out = ae(xb)
            n = min(out.shape[-1], xb.shape[-1])
            e = ((out[..., :n] - xb[..., :n]) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errs.append(e)
    return np.concatenate(errs)


def fit_threshold(err_healthy_train, k: float = 3.0) -> float:
    """Threshold = healthy-train mean + k*std."""
    return float(err_healthy_train.mean() + k * err_healthy_train.std())


def evaluate_unseen(ae, X_healthy_test, X_unseen_fault, device="cpu"):
    """ROC-AUC of healthy vs an unseen fault type, using reconstruction error as the score."""
    e_h = recon_error(ae, X_healthy_test, device)
    e_f = recon_error(ae, X_unseen_fault, device)
    scores = np.concatenate([e_h, e_f])
    labels = np.concatenate([np.zeros(len(e_h)), np.ones(len(e_f))])
    auc = roc_auc_score(labels, scores)
    log.info("unseen-fault ROC-AUC = %.3f", auc)
    return auc, e_h, e_f
