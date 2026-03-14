import torch
import torch.nn as nn

from .shape_encoder import PositionalEncoding


class BodyMotionEncoder(nn.Module):
    """Encodes per-frame body velocity ΔS_t (B, T, 16) into (B, T, d_model)."""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_ff: int = 512, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.input_proj = nn.Linear(16, d_model)
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


class HandMotionEncoder(nn.Module):
    """Encodes per-frame hand velocity ΔS_t (B, T, 84) into (B, T, d_model)."""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 dim_ff: int = 512, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.input_proj = nn.Linear(84, d_model)
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
