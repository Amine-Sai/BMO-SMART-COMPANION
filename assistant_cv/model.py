"""PyTorch 2-layer GRU sequence classifier for sign language recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


class SignNet(nn.Module):
    def __init__(self, n_cls: int, d: int = 126, h: int = 128):
        super().__init__()
        self.gru = nn.GRU(d, h, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(h, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gru(x)[0][:, -1])
