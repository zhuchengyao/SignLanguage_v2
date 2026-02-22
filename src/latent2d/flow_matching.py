from __future__ import annotations

import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowConfig2D:
    latent_dim: int = 64
    hidden: int = 512
    layers: int = 12
    # Temporal self-attention settings
    attn_heads: int = 8
    attn_every_n: int = 2  # add self-attention every N MLP blocks
    attn_dropout: float = 0.0
    # Optional per-position spatial conditioning (e.g. body latent for hand flow)
    spatial_cond_dim: int = 0


class TimestepEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor):
        # t: (B,T,1) or (B,1,1) in [0,1]
        # sinusoidal then MLP
        half = 64
        freqs = torch.exp(
            torch.linspace(0, torch.log(torch.tensor(10000.0)), half, device=t.device)
        )
        angles = t * freqs  # (B,T,half) or (B,1,half)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(pe)


class TemporalAttnBlock(nn.Module):
    """Self-attention across the temporal dimension for inter-frame coherence."""

    def __init__(self, hidden: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H)
        residual = h
        h = self.norm(h)
        h, _ = self.attn(h, h, h, need_weights=False)
        return residual + h


class MLPBlock(nn.Module):
    """Simple feed-forward block with residual."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.net(h)


class FlowUNet1D(nn.Module):
    def __init__(self, cfg: FlowConfig2D):
        super().__init__()
        D = cfg.latent_dim
        H = cfg.hidden
        self.inp = nn.Linear(D, H)

        # Build interleaved MLP + TemporalAttn blocks
        attn_every = getattr(cfg, 'attn_every_n', 2)
        attn_heads = getattr(cfg, 'attn_heads', 8)
        attn_drop = getattr(cfg, 'attn_dropout', 0.0)

        blocks = nn.ModuleList()
        for i in range(cfg.layers):
            blocks.append(MLPBlock(H))
            # Add temporal self-attention after every N MLP blocks
            if (i + 1) % attn_every == 0:
                blocks.append(TemporalAttnBlock(H, heads=attn_heads, dropout=attn_drop))
        self.blocks = blocks

        self.cond = nn.Linear(768, H)  # text/global cond: using BERT base (768)
        self.tproj = nn.Linear(128, H)
        self.out_norm = nn.LayerNorm(H)
        self.out = nn.Linear(H, D)
        self.t_embed = TimestepEmbed(128)

        # Optional spatial conditioning (per-position, e.g. body latent)
        sc_dim = getattr(cfg, 'spatial_cond_dim', 0)
        self.spatial_proj = nn.Linear(sc_dim, H) if sc_dim > 0 else None

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None,
                spatial_cond: torch.Tensor | None = None):
        # z: (B,T,D), t: (B,1,1) or (B,T,1) in [0,1]
        h = self.inp(z)
        te = self.tproj(self.t_embed(t))
        h = h + te
        if cond is not None:
            # cond: (B, Hc) -> (B,1,H) -> broadcast to (B,T,H)
            hc = self.cond(cond).unsqueeze(1).expand(-1, h.size(1), -1)
            h = h + hc
        if spatial_cond is not None and self.spatial_proj is not None:
            # spatial_cond: (B, T, spatial_cond_dim) -> (B, T, H)
            h = h + self.spatial_proj(spatial_cond)
        for blk in self.blocks:
            h = blk(h)
        v = self.out(self.out_norm(h))
        return v


def sample_flow(model: FlowUNet1D, steps: int, shape: tuple[int, int, int],
                cond: torch.Tensor | None = None,
                spatial_cond: torch.Tensor | None = None) -> torch.Tensor:
    device = next(model.parameters()).device
    B, T, D = shape
    z = torch.randn(B, T, D, device=device)
    for s in range(steps):
        t = torch.full((B, T, 1), (s + 0.5) / steps, device=device)
        v = model(z, t, cond, spatial_cond=spatial_cond)
        z = z + v / steps
    return z


@torch.no_grad()
def sample_flow_cfg(
    model: FlowUNet1D,
    steps: int,
    shape: tuple[int, int, int],
    cond: torch.Tensor,
    guidance: float = 3.0,
    uncond: torch.Tensor | None = None,
    spatial_cond: torch.Tensor | None = None,
):
    """Classifier-Free Guidance sampling on unit-time flow.

    cond: (B, Hc) text embedding
    uncond: optional (B, Hc); if None, uses zeros
    guidance: scale >= 1
    spatial_cond: optional (B, T, D_spatial) per-position conditioning
    """
    device = next(model.parameters()).device
    B, T, D = shape
    z = torch.randn(B, T, D, device=device)
    if uncond is None:
        uncond = torch.zeros_like(cond)
    for s in range(steps):
        t = torch.full((B, T, 1), (s + 0.5) / steps, device=device)
        v_c = model(z, t, cond=cond, spatial_cond=spatial_cond)
        v_u = model(z, t, cond=uncond, spatial_cond=spatial_cond)
        v = v_u + guidance * (v_c - v_u)
        z = z + v / steps
    return z



