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


class GRUClf(nn.Module):
    """GRU classifier — lighter/faster recurrent alternative to LSTM."""

    def __init__(self, n_ch: int, n_cls: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, layers, batch_first=True,
                          dropout=0.2 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):
        x = x.transpose(1, 2)              # [N, time, channels]
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class _TCNBlock(nn.Module):
    """Dilated causal residual block for the TCN."""

    def __init__(self, c_in, c_out, k, dilation, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.pad = pad
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def _crop(self, x):
        return x[..., :-self.pad] if self.pad else x      # causal: trim right padding

    def forward(self, x):
        y = self.drop(self.relu(self._crop(self.conv1(x))))
        y = self.drop(self.relu(self._crop(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return self.relu(y + res)


class TCN(nn.Module):
    """Temporal Convolutional Network — dilated causal convs, large receptive field, fast on CPU."""

    def __init__(self, n_ch: int, n_cls: int, channels=(32, 32, 64, 64), k=5):
        super().__init__()
        blocks, c_prev = [], n_ch
        for i, c in enumerate(channels):
            blocks.append(_TCNBlock(c_prev, c, k, dilation=2 ** i))
            c_prev = c
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(c_prev, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.head(self.tcn(x))


def build_model(name: str, n_ch: int, n_cls: int):
    name = name.lower()
    if name == "cnn":
        return CNN1D(n_ch, n_cls)
    if name == "lstm":
        return LSTMClf(n_ch, n_cls)
    if name == "gru":
        return GRUClf(n_ch, n_cls)
    if name == "tcn":
        return TCN(n_ch, n_cls)
    if name == "ae":
        return AE(n_ch)
    raise ValueError(f"unknown model '{name}' (cnn|lstm|gru|tcn|ae)")
