import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BodyShapeEncoder(nn.Module):
    """Encodes per-frame body joint coordinates (B, T, 16) into (B, T, d_model)."""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_ff: int = 512, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.input_proj = nn.Linear(16, d_model)  # 8 joints * 2
        self.pos_enc = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        padding_mask = ~mask if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.norm(h)


class HandShapeEncoder(nn.Module):
    """Encodes per-frame hand joint coordinates (B, T, 84) into (B, T, d_model)."""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_ff: int = 512, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.input_proj = nn.Linear(84, d_model)  # 42 joints * 2
        self.pos_enc = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        padding_mask = ~mask if mask is not None else None
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.norm(h)
