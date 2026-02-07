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
    # Diffusion decoder
    diffusion_steps: int = 200
    diffusion_sample_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2


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


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding for diffusion."""
    # t: (B,) or (B,1) int or float
    if t.dim() == 2:
        t = t.squeeze(1)
    half = dim // 2
    device = t.device
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / max(1, half))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=device)], dim=1)
    return emb


class DiffusionDecoder2D(nn.Module):
    def __init__(self, cfg: AE2DConfig):
        super().__init__()
        self.cfg = cfg
        self.num_steps = cfg.diffusion_steps
        self.out_dim = cfg.num_joints + cfg.num_joints + 2

        self.input_proj = nn.Linear(self.out_dim, cfg.d_model)
        self.cond_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.out_proj = nn.Linear(cfg.d_model, self.out_dim)

        # diffusion schedule buffers
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1.0 - alpha_bars))
        # posterior variance for DDPM sampling
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer('posterior_var', posterior_var.clamp(min=1e-8))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        # a: (T,), t: (B,)
        b = t.shape[0]
        out = a.gather(0, t).view(b, 1, 1)
        return out.expand(x_shape[0], x_shape[1], 1)

    def forward(self, noisy_y: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # noisy_y: (B,T,out_dim), t: (B,)
        h = self.input_proj(noisy_y)
        if cond is not None:
            h = h + self.cond_proj(cond)
        te = self.time_mlp(_timestep_embedding(t, self.cfg.d_model))
        h = h + te.unsqueeze(1)
        h = self.net(h)
        return self.out_proj(h)

    def q_sample(self, y0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, y0.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bars, t, y0.shape)
        return sqrt_ab * y0 + sqrt_omab * noise

    def predict_x0(self, y_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, y_t.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alpha_bars, t, y_t.shape)
        return (y_t - sqrt_omab * eps) / sqrt_ab

    @torch.no_grad()
    def p_sample(self, y_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        eps = self.forward(y_t, t, cond)
        beta_t = self._extract(self.betas, t, y_t.shape)
        alpha_t = self._extract(self.alphas, t, y_t.shape)
        alpha_bar_t = self._extract(self.alpha_bars, t, y_t.shape)
        sqrt_one_minus_ab = self._extract(self.sqrt_one_minus_alpha_bars, t, y_t.shape)
        # DDPM mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (y_t - (beta_t / sqrt_one_minus_ab) * eps)
        # noise for t>0
        noise = torch.randn_like(y_t)
        var = self._extract(self.posterior_var, t, y_t.shape)
        nonzero_mask = (t > 0).float().view(-1, 1, 1)
        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        # cond: (B,T,d_model)
        B, T, _ = cond.shape
        steps = steps or self.cfg.diffusion_sample_steps
        steps = min(steps, self.num_steps)
        # choose timesteps uniformly
        idx = torch.linspace(0, self.num_steps - 1, steps).round().long().to(cond.device)
        idx = torch.flip(idx, dims=[0])  # descending
        y = torch.randn(B, T, self.out_dim, device=cond.device)
        for t in idx:
            t_batch = torch.full((B,), int(t.item()), device=cond.device, dtype=torch.long)
            y = self.p_sample(y, t_batch, cond)
        return y

class AutoEncoder2D(nn.Module):
    def __init__(self, cfg: AE2DConfig):
        super().__init__()
        # Backward compatibility for older checkpoints
        if not hasattr(cfg, 'diffusion_steps'):
            cfg.diffusion_steps = 200
        if not hasattr(cfg, 'diffusion_sample_steps'):
            cfg.diffusion_sample_steps = 50
        if not hasattr(cfg, 'beta_start'):
            cfg.beta_start = 1e-4
        if not hasattr(cfg, 'beta_end'):
            cfg.beta_end = 2e-2
        self.cfg = cfg
        self.encoder = TemporalEncoder2D(cfg)
        self.decoder = DiffusionDecoder2D(cfg)

    def forward(self, xy_seq: torch.Tensor) -> torch.Tensor:
        # Sampling-based decode for inference
        memory = self.encoder(xy_seq)
        preds = self.decoder.sample(memory, steps=self.cfg.diffusion_sample_steps)
        return preds  # (B, T, 50+50+2)

    def diffusion_loss(
        self,
        xy_seq: torch.Tensor,
        target_params: torch.Tensor,
        weights: torch.Tensor | None = None,
        vel_weight: float = 0.0,
        acc_weight: float = 0.0,
    ) -> torch.Tensor:
        # xy_seq: (B,T,100), target_params: (B,T,out_dim)
        memory = self.encoder(xy_seq)
        B = target_params.size(0)
        t = torch.randint(0, self.decoder.num_steps, (B,), device=xy_seq.device)
        noise = torch.randn_like(target_params)
        noisy = self.decoder.q_sample(target_params, t, noise)
        pred_noise = self.decoder(noisy, t, cond=memory)
        mse = (pred_noise - noise) ** 2
        if weights is not None:
            mse = mse * weights.view(1, 1, -1)
        loss = mse.mean()
        if vel_weight > 0.0 or acc_weight > 0.0:
            with torch.no_grad():
                x0_pred = self.decoder.predict_x0(noisy, t, pred_noise)
            J = self.cfg.num_joints
            pred_theta = x0_pred[..., :J]
            pred_root = x0_pred[..., 2*J:2*J+2]
            if vel_weight > 0.0:
                dtheta = pred_theta[:, 1:] - pred_theta[:, :-1]
                droot = pred_root[:, 1:] - pred_root[:, :-1]
                loss = loss + vel_weight * 0.5 * (dtheta.abs().mean() + droot.abs().mean())
            if acc_weight > 0.0 and pred_theta.size(1) >= 3:
                ddtheta = pred_theta[:, 2:] - 2 * pred_theta[:, 1:-1] + pred_theta[:, :-2]
                ddroot = pred_root[:, 2:] - 2 * pred_root[:, 1:-1] + pred_root[:, :-2]
                loss = loss + acc_weight * 0.5 * (ddtheta.abs().mean() + ddroot.abs().mean())
        return loss


