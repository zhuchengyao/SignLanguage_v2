from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.device)


@dataclass
class AE2DConfig:
    pose_dim_xy: int = 100            # 50 joints * (x,y)
    num_joints: int = 50
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_ff: int = 256
    dropout: float = 0.1
    max_len: int = 128


class TemporalEncoder2D(nn.Module):
    def __init__(self, cfg: AE2DConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.pose_dim_xy, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 100)
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        return self.norm(h)


class TemporalDecoder2D(nn.Module):
    def __init__(self, cfg: AE2DConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.num_layers)
        # Per-frame outputs: local_theta(J), bone_scales(J), root_xy(2)
        out_dim = cfg.num_joints + cfg.num_joints + 2
        self.head = nn.Linear(cfg.d_model, out_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        # memory: (B, T, D)
        tgt = self.input_proj(memory)
        tgt = self.pos(tgt)
        out = self.decoder(tgt=tgt, memory=memory)
        return self.head(out)


class AutoEncoder2D(nn.Module):
    def __init__(self, cfg: AE2DConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TemporalEncoder2D(cfg)
        self.decoder = TemporalDecoder2D(cfg)

    def forward(self, xy_seq: torch.Tensor) -> torch.Tensor:
        # xy_seq: (B, T, 100)
        memory = self.encoder(xy_seq)
        preds = self.decoder(memory)
        return preds  # (B, T, 50+50+2)


