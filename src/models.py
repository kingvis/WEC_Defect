"""Light, CPU-friendly models: 1D-CNN, LSTM classifier, conv autoencoder."""
from __future__ import annotations
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Compact 1D-CNN for fault detection / classification. Good at local signal motifs."""

    def __init__(self, n_ch: int, n_cls: int, k: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, k, padding=k // 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, k, padding=k // 2),   nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, k, padding=k // 2),   nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.net(x)


class LSTMClf(nn.Module):
    """LSTM for temporal evolution. Expects [N, channels, time] -> transposes internally."""

    def __init__(self, n_ch: int, n_cls: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, layers, batch_first=True,
                            dropout=0.2 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):
        x = x.transpose(1, 2)              # [N, time, channels]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])    # last time step


class AE(nn.Module):
    """1D conv autoencoder for UNSUPERVISED anomaly detection (train on healthy only)."""

    def __init__(self, n_ch: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_ch, 32, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(32, 16, 7, 2, 3), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(16, 32, 7, 2, 3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, n_ch, 7, 2, 3, output_padding=1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def build_model(name: str, n_ch: int, n_cls: int):
    name = name.lower()
    if name == "cnn":
        return CNN1D(n_ch, n_cls)
    if name == "lstm":
        return LSTMClf(n_ch, n_cls)
    if name == "ae":
        return AE(n_ch)
    raise ValueError(f"unknown model '{name}' (cnn|lstm|ae)")
