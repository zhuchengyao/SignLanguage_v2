"""Motion VAE: continuous latent space autoencoder for pose sequences.

Replaces VQ-VAE's discrete codebook with a Gaussian VAE bottleneck (mu, logvar).
Reuses MotionEncoder and MotionDecoder from model_vqvae.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from .config import T2M_Config
from .model_vqvae import MotionEncoder, MotionDecoder, PositionalEncoding
from .latent2d.skeleton2d import get_body_connections, get_hand_connections


class VAEBottleneck(nn.Module):
    """Variational bottleneck: maps encoder output to mu/logvar, reparameterizes."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: (B, T_down, input_dim) encoder output
        Returns:
            z: (B, T_down, latent_dim) sampled latent
            mu: (B, T_down, latent_dim)
            logvar: (B, T_down, latent_dim)
        """
        mu = self.fc_mu(z_e)
        logvar = self.fc_logvar(z_e)
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # deterministic at eval
        return z, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,1)), averaged over all elements."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class MotionVAE(nn.Module):
    """Motion VAE with Transformer encoder/decoder and Gaussian latent space."""

    def __init__(self, cfg: T2M_Config):
        super().__init__()
        self.cfg = cfg
        latent_dim = cfg.vae_latent_dim

        # Encoder: pose -> embedding_dim (reuse from VQ-VAE)
        self.encoder = MotionEncoder(cfg)

        # VAE bottleneck: embedding_dim -> latent_dim
        self.bottleneck = VAEBottleneck(cfg.embedding_dim, latent_dim)

        # Decoder: needs input_projection from latent_dim (not embedding_dim)
        # We override the decoder's input projection dimension
        self.decoder = MotionDecoder(cfg)
        # Replace the decoder's input projection to accept latent_dim instead of embedding_dim
        self.decoder.input_projection = nn.Linear(latent_dim, cfg.motion_decoder_hidden_dim)

    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode pose sequence to latent space.

        Returns:
            z: (B, T_down, latent_dim)
            mu: (B, T_down, latent_dim)
            logvar: (B, T_down, latent_dim)
        """
        z_e = self.encoder(x, mask)
        z, mu, logvar = self.bottleneck(z_e)
        return z, mu, logvar

    def decode(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        """Decode latent to pose sequence.

        Args:
            z: (B, T_down, latent_dim)
            target_length: original sequence length
        Returns:
            reconstructed: (B, target_length, pose_dim)
        """
        return self.decoder(z, target_length)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
        B, T, D = x.shape

        # Encode
        z_e = self.encoder(x, mask)
        z, mu, logvar = self.bottleneck(z_e)

        # KL divergence
        kl_loss = VAEBottleneck.kl_divergence(mu, logvar)

        # Decode
        target_length = int(mask.sum(dim=1).max().item())
        reconstructed_short = self.decoder(z, target_length)

        if target_length < T:
            reconstructed = F.pad(reconstructed_short, (0, 0, 0, T - target_length))
        else:
            reconstructed = reconstructed_short

        # --- Reconstruction loss ---
        valid_positions = mask.unsqueeze(-1).float()
        recon_loss = F.mse_loss(reconstructed * valid_positions, x * valid_positions, reduction='sum')
        recon_loss = recon_loss / valid_positions.sum().clamp(min=1.0)

        # --- Sign-aware extras: weighted recon, bone-length, temporal smoothness ---
        def split_xy_conf(tensor_150: torch.Tensor):
            body = tensor_150[..., :24].view(*tensor_150.shape[:-1], 8, 3)
            right = tensor_150[..., 24:87].view(*tensor_150.shape[:-1], 21, 3)
            left = tensor_150[..., 87:150].view(*tensor_150.shape[:-1], 21, 3)
            body_xy, body_c = body[..., :2], body[..., 2]
            right_xy, right_c = right[..., :2], right[..., 2]
            left_xy, left_c = left[..., :2], left[..., 2]
            return (body_xy, right_xy, left_xy), (body_c, right_c, left_c)

        # Denormalize if stats available
        x_denorm, rec_denorm = x, reconstructed
        if self.cfg.mean is not None and self.cfg.std is not None:
            mean = self.cfg.mean.to(x.device)
            std = self.cfg.std.to(x.device)
            x_denorm = x * std + mean
            rec_denorm = reconstructed * std + mean

        (bx, rx, lx), (bc, rc, lc) = split_xy_conf(x_denorm)
        (br, rr, lr), _ = split_xy_conf(rec_denorm)

        body_w = torch.full_like(bc, fill_value=self.cfg.body_point_weight)
        hand_w = torch.full_like(rc, fill_value=self.cfg.hand_point_weight)
        if self.cfg.use_confidence_weight:
            gamma = self.cfg.conf_weight_gamma
            body_w = body_w * bc.clamp(min=0.0, max=1.0).pow(gamma)
            hand_w_r = hand_w * rc.clamp(min=0.0, max=1.0).pow(gamma)
            hand_w_l = hand_w * lc.clamp(min=0.0, max=1.0).pow(gamma)
        else:
            hand_w_r = hand_w
            hand_w_l = hand_w

        vmask_t = valid_positions.bool()

        def mse_weighted(a, b, w, vm):
            diff2 = (a - b).pow(2).sum(dim=-1)
            vm2 = vm if vm.dim() == 3 else vm.unsqueeze(-1)
            w2 = w * vm2.to(w.dtype)
            loss = (diff2 * w2).sum()
            denom = w2.sum().clamp(min=1.0)
            return loss / denom

        recon_weighted = (mse_weighted(br, bx, body_w, vmask_t)
                          + mse_weighted(rr, rx, hand_w_r, vmask_t)
                          + mse_weighted(lr, lx, hand_w_l, vmask_t))

        def bone_length_loss(xy_pred, xy_gt, connections, vm):
            bl = 0.0
            denom = 0.0
            for a, b in connections:
                pa, pb = xy_pred[..., a, :], xy_pred[..., b, :]
                ga, gb = xy_gt[..., a, :], xy_gt[..., b, :]
                len_p = (pa - pb).norm(dim=-1)
                len_g = (ga - gb).norm(dim=-1).clamp(min=1e-6)
                bl += ((len_p - len_g).abs() * vm.squeeze(-1)).sum()
                denom += vm.sum().item()
            return bl / max(denom, 1.0)

        bone_loss = bone_length_loss(br, bx, get_body_connections(), vmask_t)
        bone_loss += bone_length_loss(rr, rx, get_hand_connections(), vmask_t)
        bone_loss += bone_length_loss(lr, lx, get_hand_connections(), vmask_t)

        def temporal_smooth_loss(xy_seq, vm, order=1):
            if xy_seq.size(1) < (order + 1):
                return torch.tensor(0.0, device=xy_seq.device, dtype=xy_seq.dtype)
            if order == 1:
                d = xy_seq[:, 1:] - xy_seq[:, :-1]
                vmask = vm[:, 1:] * vm[:, :-1]
            else:
                d1 = xy_seq[:, 1:] - xy_seq[:, :-1]
                d = d1[:, 1:] - d1[:, :-1]
                vmask = vm[:, 2:] * vm[:, 1:-1] * vm[:, :-2]
            loss = d.abs().sum(dim=-1)
            vmask_k = vmask if vmask.dim() == 3 else vmask.unsqueeze(-1)
            loss = (loss * vmask_k.to(loss.dtype)).sum()
            return loss / vmask_k.sum().clamp(min=1.0)

        vel_loss = temporal_smooth_loss(torch.cat([br, rr, lr], dim=2), vmask_t, order=1)
        acc_loss = temporal_smooth_loss(torch.cat([br, rr, lr], dim=2), vmask_t, order=2)

        return {
            'reconstructed': reconstructed,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'recon_weighted': recon_weighted,
            'bone_length_loss': bone_loss,
            'velocity_loss': vel_loss,
            'accel_loss': acc_loss,
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }
