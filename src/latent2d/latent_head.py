from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .model_ae2d import AutoEncoder2D, AE2DConfig


@dataclass
class LatentConfig2D:
    latent_dim: int = 64


class LatentBottleneck2D(nn.Module):
    def __init__(self, ae_cfg: AE2DConfig, lat_cfg: LatentConfig2D):
        super().__init__()
        self.ae = AutoEncoder2D(ae_cfg)
        self.to_latent = nn.Linear(ae_cfg.d_model, lat_cfg.latent_dim)
        self.from_latent = nn.Linear(lat_cfg.latent_dim, ae_cfg.d_model)
        self.latent_dim = lat_cfg.latent_dim
        self.ae_cfg = ae_cfg

    @torch.no_grad()
    def encode_to_latent(self, xy_seq: torch.Tensor) -> torch.Tensor:
        # xy_seq: (B,T,100)
        memory = self.ae.encoder(xy_seq)  # (B,T,d_model)
        z = self.to_latent(memory)        # (B,T,latent_dim)
        return z

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,T,latent_dim)
        memory = self.from_latent(z)
        preds = self.ae.decoder(memory)  # (B,T, 50+50+2)
        return preds



