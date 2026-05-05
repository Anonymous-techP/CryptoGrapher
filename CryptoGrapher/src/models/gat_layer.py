"""Graph attention layer + correlation-induced graph builder."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        N = Wh.size(0)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1
        )
        return all_combinations.view(N, N, 2 * self.out_features)


def build_correlation_matrix(price_data: np.ndarray,
                             threshold: float = 0.5) -> np.ndarray:
    corr_matrix = np.corrcoef(price_data)
    relation_matrix = np.where(
        np.abs(corr_matrix) > threshold, np.abs(corr_matrix), 0.0
    )
    np.fill_diagonal(relation_matrix, 1.0)
    return relation_matrix
