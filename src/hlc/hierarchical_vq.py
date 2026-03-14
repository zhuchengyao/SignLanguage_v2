from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQCodebook(nn.Module):
    """Single vector-quantization codebook with EMA updates."""

    def __init__(self, codebook_size: int, embedding_dim: int,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())
        self.register_buffer("usage_count", torch.zeros(codebook_size))

    def _find_nearest(self, flat: torch.Tensor) -> torch.Tensor:
        emb = self.embedding.weight
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            + emb.pow(2).sum(1)
            - 2 * flat @ emb.t()
        )
        return torch.argmin(dist, dim=1)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        z_e: (B, T, D) continuous encoder output.
        Returns: quantized (B, T, D), indices (B, T), loss dict.
        """
        B, T, D = z_e.shape
        flat = z_e.reshape(-1, D)

        with torch.no_grad():
            indices = self._find_nearest(flat)

        quantized = self.embedding(indices).view(B, T, D)

        if self.training:
            self.usage_count.index_add_(
                0, indices, torch.ones_like(indices, dtype=torch.float)
            )
            encodings = F.one_hot(indices, self.codebook_size).float()
            self.ema_cluster_size.mul_(self.ema_decay).add_(
                encodings.sum(0), alpha=1 - self.ema_decay
            )
            dw = encodings.t() @ flat
            self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon) * n
            )
            self.embedding.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), z_e)
        vq_loss = self.commitment_cost * e_latent_loss

        quantized_st = z_e + (quantized - z_e).detach()
        return quantized_st, indices.view(B, T), {
            "vq_loss": vq_loss,
            "commitment_loss": e_latent_loss,
        }

    @torch.no_grad()
    def reinit_dead_codes(self, z_e: torch.Tensor) -> int:
        if not self.training:
            return 0
        dead = torch.where(self.usage_count == 0)[0]
        if dead.numel() == 0:
            return 0
        flat = z_e.reshape(-1, self.embedding_dim)
        rand_idx = torch.randint(0, flat.size(0), (dead.numel(),), device=z_e.device)
        new_vecs = flat[rand_idx]
        self.embedding.weight.data[dead] = new_vecs
        self.ema_w.data[dead] = new_vecs
        self.ema_cluster_size[dead] = 1.0
        self.usage_count.zero_()
        return dead.numel()


class HierarchicalVQ(nn.Module):
    """Four independent codebooks: global/local x shape/motion."""

    def __init__(self, global_codebook_size: int = 512,
                 local_codebook_size: int = 1024,
                 embedding_dim: int = 256,
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.99):
        super().__init__()
        self.global_shape_vq = VQCodebook(
            global_codebook_size, embedding_dim, commitment_cost, ema_decay
        )
        self.local_shape_vq = VQCodebook(
            local_codebook_size, embedding_dim, commitment_cost, ema_decay
        )
        self.global_motion_vq = VQCodebook(
            global_codebook_size, embedding_dim, commitment_cost, ema_decay
        )
        self.local_motion_vq = VQCodebook(
            local_codebook_size, embedding_dim, commitment_cost, ema_decay
        )

    def forward(
        self,
        body_shape_feat: torch.Tensor,
        hand_shape_feat: torch.Tensor,
        body_motion_feat: torch.Tensor,
        hand_motion_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor | dict]:
        """
        Each input: (B, T, D).
        Returns dict with quantized tensors, indices, and aggregated losses.
        """
        gs_q, gs_idx, gs_loss = self.global_shape_vq(body_shape_feat)
        ls_q, ls_idx, ls_loss = self.local_shape_vq(hand_shape_feat)
        gm_q, gm_idx, gm_loss = self.global_motion_vq(body_motion_feat)
        lm_q, lm_idx, lm_loss = self.local_motion_vq(hand_motion_feat)

        total_vq = (
            gs_loss["vq_loss"] + ls_loss["vq_loss"]
            + gm_loss["vq_loss"] + lm_loss["vq_loss"]
        )
        total_commit = (
            gs_loss["commitment_loss"] + ls_loss["commitment_loss"]
            + gm_loss["commitment_loss"] + lm_loss["commitment_loss"]
        )

        return {
            "quantized": {
                "global_shape": gs_q,
                "local_shape": ls_q,
                "global_motion": gm_q,
                "local_motion": lm_q,
            },
            "indices": {
                "global_shape": gs_idx,
                "local_shape": ls_idx,
                "global_motion": gm_idx,
                "local_motion": lm_idx,
            },
            "vq_loss": total_vq,
            "commitment_loss": total_commit,
        }

    def embed_indices(
        self,
        gs_idx: torch.Tensor,
        ls_idx: torch.Tensor,
        gm_idx: torch.Tensor,
        lm_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "global_shape": self.global_shape_vq.embedding(gs_idx),
            "local_shape": self.local_shape_vq.embedding(ls_idx),
            "global_motion": self.global_motion_vq.embedding(gm_idx),
            "local_motion": self.local_motion_vq.embedding(lm_idx),
        }

    def reinit_dead_codes(
        self,
        body_shape_feat: torch.Tensor,
        hand_shape_feat: torch.Tensor,
        body_motion_feat: torch.Tensor,
        hand_motion_feat: torch.Tensor,
    ) -> int:
        n = 0
        n += self.global_shape_vq.reinit_dead_codes(body_shape_feat)
        n += self.local_shape_vq.reinit_dead_codes(hand_shape_feat)
        n += self.global_motion_vq.reinit_dead_codes(body_motion_feat)
        n += self.local_motion_vq.reinit_dead_codes(hand_motion_feat)
        return n
