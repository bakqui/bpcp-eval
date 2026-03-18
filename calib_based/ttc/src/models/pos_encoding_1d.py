import math
import torch
import torch.nn as nn


class FixedPositionalEncoding1D(nn.Module):
    """Non-learnable 1D sine-cosine positional encoding (ViT-style)."""
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):  # x: (B,N,D)
        N = x.size(1)
        return x + self.pe[:, :N, :]
