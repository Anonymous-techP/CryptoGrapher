"""CryptoGrapher: LSTM encoder + GAT over a correlation-induced graph."""
from __future__ import annotations

import torch
import torch.nn as nn

from .gat_layer import GATLayer


class CryptoGrapher(nn.Module):
    def __init__(self,
                 d_feat: int = 5,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 num_graph_layer: int = 2):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_size, hidden_size, dropout=dropout)
            for _ in range(num_graph_layer)
        ])

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self,
                x: torch.Tensor,
                relation_matrix=None) -> torch.Tensor:
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]

        if relation_matrix is None:
            return self.fc(hidden)

        for gat in self.gat_layers:
            hidden = gat(hidden, relation_matrix)

        return self.fc(hidden)
