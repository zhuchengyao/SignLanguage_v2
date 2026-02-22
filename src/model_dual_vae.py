"""Dual-branch Motion VAE: decoupled body + hand modeling.

Body branch (24-dim, 8 joints): low-frequency, stronger temporal smoothness.
Hand branch (126-dim, 42 joints): high-frequency, higher reconstruction fidelity.
Hand decoder is conditioned on body latent for wrist synchronization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any

from .config import T2M_Config
from .model_vae import VAEBottleneck


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :].to(x.device)


class BranchEncoder(nn.Module):
    """Transformer encoder for a single pose branch."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 num_layers: int, num_heads: int, downsample_rate: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.temporal_downsample = nn.Conv1d(
            hidden_dim, embedding_dim,
            kernel_size=downsample_rate, stride=downsample_rate,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)
        x = x.transpose(1, 2)
        x = self.temporal_downsample(x)
        return x.transpose(1, 2)


class BranchDecoder(nn.Module):
    """Transformer decoder for a single pose branch."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, num_heads: int, downsample_rate: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.temporal_upsample = nn.ConvTranspose1d(
            hidden_dim, hidden_dim,
            kernel_size=downsample_rate, stride=downsample_rate,
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        x = self.input_projection(z)
        x = x.transpose(1, 2)
        x = self.temporal_upsample(x)
        x = x.transpose(1, 2)
        x = x[:, :target_length, :]
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.final_norm(x)
        return self.output_projection(x)


class DualMotionVAE(nn.Module):
    """Dual-branch VAE with separate body and hand pathways."""

    def __init__(self, cfg: T2M_Config):
        super().__init__()
        self.cfg = cfg
        ds = cfg.downsample_rate
        max_len = cfg.model_max_seq_len
        emb = cfg.embedding_dim

        body_h = cfg.dual_body_hidden_dim
        hand_h = cfg.dual_hand_hidden_dim
        body_lat = cfg.dual_body_latent_dim
        hand_lat = cfg.dual_hand_latent_dim
        n_layers = cfg.dual_enc_layers
        n_heads = cfg.dual_heads
        drop = cfg.dual_dropout

        # ---- Body branch (24-dim) ----
        self.body_encoder = BranchEncoder(24, body_h, emb, n_layers, n_heads, ds, max_len, drop)
        self.body_bottleneck = VAEBottleneck(emb, body_lat)
        self.body_decoder = BranchDecoder(body_lat, body_h, 24, n_layers, n_heads, ds, max_len, drop)

        # ---- Hand branch (126-dim), decoder conditioned on body latent ----
        self.hand_encoder = BranchEncoder(126, hand_h, emb, n_layers, n_heads, ds, max_len, drop)
        self.hand_bottleneck = VAEBottleneck(emb, hand_lat)
        self.hand_decoder = BranchDecoder(
            hand_lat + body_lat, hand_h, 126, n_layers, n_heads, ds, max_len, drop,
        )

    # ---- pose splitting / merging ----

    @staticmethod
    def split_pose(x: torch.Tensor):
        """150-dim -> body(24), hands(126) where hands = right(63) || left(63)."""
        body = x[..., :24]
        hands = torch.cat([x[..., 24:87], x[..., 87:150]], dim=-1)
        return body, hands

    @staticmethod
    def merge_pose(body: torch.Tensor, hands: torch.Tensor) -> torch.Tensor:
        """body(24) + hands(126) -> 150-dim."""
        return torch.cat([body, hands[..., :63], hands[..., 63:]], dim=-1)

    # ---- encode / decode API ----

    def encode_body(self, x_body, mask=None):
        z_e = self.body_encoder(x_body, mask)
        return self.body_bottleneck(z_e)

    def encode_hand(self, x_hand, mask=None):
        z_e = self.hand_encoder(x_hand, mask)
        return self.hand_bottleneck(z_e)

    def decode_body(self, z_body, target_length):
        return self.body_decoder(z_body, target_length)

    def decode_hand(self, z_hand, z_body, target_length):
        z_cat = torch.cat([z_hand, z_body], dim=-1)
        return self.hand_decoder(z_cat, target_length)

    def decode_full(self, z_body, z_hand, target_length):
        """Decode both branches and merge to 150-dim."""
        body_out = self.decode_body(z_body, target_length)
        hand_out = self.decode_hand(z_hand, z_body, target_length)
        return self.merge_pose(body_out, hand_out)

    # ---- full forward with loss computation ----

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
        B, T, _ = x.shape
        body_gt, hands_gt = self.split_pose(x)

        # Encode
        z_body, mu_b, logvar_b = self.encode_body(body_gt, mask)
        z_hand, mu_h, logvar_h = self.encode_hand(hands_gt, mask)

        # KL
        kl_body = VAEBottleneck.kl_divergence(mu_b, logvar_b)
        kl_hand = VAEBottleneck.kl_divergence(mu_h, logvar_h)

        # Decode
        tgt_len = int(mask.sum(dim=1).max().item())
        body_rec = self.decode_body(z_body, tgt_len)
        hand_rec = self.decode_hand(z_hand, z_body.detach(), tgt_len)

        # Pad to original T if needed
        if tgt_len < T:
            body_rec = F.pad(body_rec, (0, 0, 0, T - tgt_len))
            hand_rec = F.pad(hand_rec, (0, 0, 0, T - tgt_len))

        valid = mask.unsqueeze(-1).float()  # (B, T, 1)

        # ---- Reconstruction losses (masked MSE) ----
        def masked_mse(pred, gt):
            return F.mse_loss(pred * valid, gt * valid, reduction='sum') / valid.sum().clamp(min=1.0)

        body_recon_loss = masked_mse(body_rec, body_gt)
        hand_recon_loss = masked_mse(hand_rec, hands_gt)

        # ---- Wrist synchronization loss ----
        # Body: joint 4 = right wrist, joint 7 = left wrist (each joint = 3 dims: x,y,conf)
        # Hands: first joint of each hand is the root (wrist)
        body_rwrist_xy = body_rec[..., 12:14]   # joint4 x,y
        body_lwrist_xy = body_rec[..., 21:23]   # joint7 x,y
        hand_rroot_xy = hand_rec[..., 0:2]      # right hand root x,y
        hand_lroot_xy = hand_rec[..., 63:65]    # left hand root x,y

        sync_loss = (
            masked_mse(body_rwrist_xy, hand_rroot_xy)
            + masked_mse(body_lwrist_xy, hand_lroot_xy)
        )

        # ---- Temporal smoothness ----
        vm = valid.bool()

        def temporal_loss(seq, order=1):
            if seq.size(1) < order + 1:
                return seq.new_tensor(0.0)
            if order == 1:
                d = seq[:, 1:] - seq[:, :-1]
                m = (vm[:, 1:] * vm[:, :-1]).squeeze(-1).float()  # (B, T-1)
            else:
                d1 = seq[:, 1:] - seq[:, :-1]
                d = d1[:, 1:] - d1[:, :-1]
                m = (vm[:, 2:] * vm[:, 1:-1] * vm[:, :-2]).squeeze(-1).float()
            loss_val = d.abs().sum(dim=-1)  # (B, T-k)
            return (loss_val * m).sum() / m.sum().clamp(min=1.0)

        body_vel = temporal_loss(body_rec, order=1)
        body_acc = temporal_loss(body_rec, order=2)
        hand_vel = temporal_loss(hand_rec, order=1)
        hand_acc = temporal_loss(hand_rec, order=2)

        reconstructed = self.merge_pose(body_rec, hand_rec)

        return {
            'reconstructed': reconstructed,
            'body_recon_loss': body_recon_loss,
            'hand_recon_loss': hand_recon_loss,
            'kl_body': kl_body,
            'kl_hand': kl_hand,
            'sync_loss': sync_loss,
            'body_vel': body_vel,
            'body_acc': body_acc,
            'hand_vel': hand_vel,
            'hand_acc': hand_acc,
            'z_body': z_body, 'z_hand': z_hand,
            'mu_body': mu_b, 'mu_hand': mu_h,
        }
