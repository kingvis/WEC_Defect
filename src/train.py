"""Classifier training loop with class-weighting for imbalance + early stopping."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from .utils import OUTPUTS, get_logger

log = get_logger("train")


def train(model, ds_tr, ds_val, n_cls, y_tr, epochs: int = 40, lr: float = 1e-3,
          bs: int = 64, device: str = "cpu", ckpt: str = None, patience: int = 6):
    """Train a classifier. Saves the best (lowest val-loss) checkpoint and reloads it."""
    ckpt = ckpt or str(OUTPUTS / "best.pt")
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    classes = np.arange(n_cls)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_va = DataLoader(ds_val, batch_size=bs)

    best, wait = float("inf"), 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                vloss += crit(model(xb.to(device)), yb.to(device)).item()
        vloss /= max(1, len(dl_va))
        log.info("epoch %d  val_loss %.4f", ep, vloss)

        if vloss < best:
            best, wait = vloss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= patience:
                log.info("early stop at epoch %d (best val_loss %.4f)", ep, best)
                break

    model.load_state_dict(torch.load(ckpt))
    return model
