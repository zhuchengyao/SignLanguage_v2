from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowConfig2D:
    latent_dim: int = 64
    hidden: int = 512
    layers: int = 12


class TimestepEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor):
        # t: (B,T,1) in [0,1]
        # sinusoidal then MLP
        half = 64
        freqs = torch.exp(
            torch.linspace(0, torch.log(torch.tensor(10000.0)), half, device=t.device)
        )
        angles = t * freqs  # (B,T,half)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(pe)


class FlowUNet1D(nn.Module):
    def __init__(self, cfg: FlowConfig2D):
        super().__init__()
        D = cfg.latent_dim
        H = cfg.hidden
        self.inp = nn.Linear(D, H)
        blocks = []
        for _ in range(cfg.layers):
            blocks += [nn.Sequential(nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, H))]
        self.blocks = nn.ModuleList(blocks)
        self.cond = nn.Linear(768, H)  # text/global cond: using BERT base (768)
        self.tproj = nn.Linear(128, H)
        self.out = nn.Linear(H, D)
        self.t_embed = TimestepEmbed(128)

    def forward(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None):
        # z: (B,T,D), t: (B,T,1) in [0,1]
        h = self.inp(z)
        te = self.tproj(self.t_embed(t))
        h = h + te
        if cond is not None:
            # cond: (B, Hc) -> (B,1,H) -> (B,T,H)
            hc = self.cond(cond).unsqueeze(1).expand(-1, h.size(1), -1)
            h = h + hc
        for blk in self.blocks:
            h = h + blk(h)
        v = self.out(h)
        return v


def sample_flow(model: FlowUNet1D, steps: int, shape: tuple[int, int, int], cond: torch.Tensor | None = None) -> torch.Tensor:
    # simple Euler sampler on unit-time flow
    device = next(model.parameters()).device
    B, T, D = shape
    z = torch.randn(B, T, D, device=device)
    for s in range(steps):
        t = torch.full((B, T, 1), (s + 0.5) / steps, device=device)
        v = model(z, t, cond)
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
):
    """Classifier-Free Guidance sampling on unit-time flow.

    cond: (B, Hc) text embedding
    uncond: optional (B, Hc); if None, uses zeros
    guidance: scale >= 1
    """
    device = next(model.parameters()).device
    B, T, D = shape
    z = torch.randn(B, T, D, device=device)
    if uncond is None:
        uncond = torch.zeros_like(cond)
    for s in range(steps):
        t = torch.full((B, T, 1), (s + 0.5) / steps, device=device)
        v_c = model(z, t, cond=cond)
        v_u = model(z, t, cond=uncond)
        v = v_u + guidance * (v_c - v_u)
        z = z + v / steps
    return z



