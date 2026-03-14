from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..config_hlc import HLC_NAR_Config
from .shape_encoder import BodyShapeEncoder, HandShapeEncoder
from .motion_encoder import BodyMotionEncoder, HandMotionEncoder
from .hierarchical_vq import HierarchicalVQ


def split_body_hand_xy(pose_150: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split 150-dim pose into body_xy (B,T,16) and hand_xy (B,T,84).

    Layout: body 8*3 = 24, right_hand 21*3 = 63, left_hand 21*3 = 63.
    We extract xy (drop confidence) then flatten.
    """
    body_raw = pose_150[..., :24].reshape(*pose_150.shape[:-1], 8, 3)
    right_raw = pose_150[..., 24:87].reshape(*pose_150.shape[:-1], 21, 3)
    left_raw = pose_150[..., 87:150].reshape(*pose_150.shape[:-1], 21, 3)

    body_xy = body_raw[..., :2].reshape(*pose_150.shape[:-1], 16)
    right_xy = right_raw[..., :2]
    left_xy = left_raw[..., :2]
    hand_xy = torch.cat([left_xy, right_xy], dim=-2)  # (..., 42, 2)
    hand_xy = hand_xy.reshape(*pose_150.shape[:-1], 84)

    return body_xy, hand_xy


def compute_delta(seq: torch.Tensor, mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Compute frame differences ΔS_t = S_t - S_{t-1}.

    Returns delta (B, T-1, D) and trimmed mask (B, T-1) if provided.
    """
    delta = seq[:, 1:] - seq[:, :-1]
    delta_mask = None
    if mask is not None:
        delta_mask = mask[:, 1:] & mask[:, :-1]
    return delta, delta_mask


class HLCModule(nn.Module):
    """Hierarchical Latent Codebook module.

    Two parallel encoding branches (shape + motion) with hierarchical VQ
    (global body + local hand) for each branch.
    """

    def __init__(self, cfg: HLC_NAR_Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.hlc_d_model

        self.body_shape_enc = BodyShapeEncoder(
            d_model=d, nhead=cfg.hlc_nhead, num_layers=cfg.hlc_num_layers,
            dim_ff=cfg.hlc_dim_ff, dropout=cfg.hlc_dropout, max_len=cfg.max_seq_len,
        )
        self.hand_shape_enc = HandShapeEncoder(
            d_model=d, nhead=cfg.hlc_nhead, num_layers=cfg.hlc_num_layers,
            dim_ff=cfg.hlc_dim_ff, dropout=cfg.hlc_dropout, max_len=cfg.max_seq_len,
        )
        self.body_motion_enc = BodyMotionEncoder(
            d_model=d, nhead=cfg.hlc_nhead, num_layers=cfg.hlc_num_layers,
            dim_ff=cfg.hlc_dim_ff, dropout=cfg.hlc_dropout, max_len=cfg.max_seq_len,
        )
        self.hand_motion_enc = HandMotionEncoder(
            d_model=d, nhead=cfg.hlc_nhead, num_layers=cfg.hlc_num_layers,
            dim_ff=cfg.hlc_dim_ff, dropout=cfg.hlc_dropout, max_len=cfg.max_seq_len,
        )
        self.vq = HierarchicalVQ(
            global_codebook_size=cfg.global_codebook_size,
            local_codebook_size=cfg.local_codebook_size,
            embedding_dim=cfg.vq_embedding_dim,
            commitment_cost=cfg.vq_commitment_cost,
            ema_decay=cfg.vq_ema_decay,
        )

    def encode(
        self,
        pose_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | dict]:
        """
        pose_seq: (B, T, 150) normalised pose.
        mask: (B, T) bool, True for valid frames.
        Returns dict with quantized embeddings, indices, losses.
        """
        body_xy, hand_xy = split_body_hand_xy(pose_seq)

        body_delta, delta_mask = compute_delta(body_xy, mask)
        hand_delta, _ = compute_delta(hand_xy, mask)

        body_shape_feat = self.body_shape_enc(body_xy, mask)
        hand_shape_feat = self.hand_shape_enc(hand_xy, mask)
        body_motion_feat = self.body_motion_enc(body_delta, delta_mask)
        hand_motion_feat = self.hand_motion_enc(hand_delta, delta_mask)

        # Align motion features to shape length by prepending a zero frame
        B = body_motion_feat.size(0)
        zero_pad = torch.zeros(B, 1, body_motion_feat.size(2),
                               device=body_motion_feat.device, dtype=body_motion_feat.dtype)
        body_motion_feat = torch.cat([zero_pad, body_motion_feat], dim=1)
        hand_motion_feat = torch.cat([
            torch.zeros(B, 1, hand_motion_feat.size(2),
                        device=hand_motion_feat.device, dtype=hand_motion_feat.dtype),
            hand_motion_feat,
        ], dim=1)

        vq_out = self.vq(body_shape_feat, hand_shape_feat,
                         body_motion_feat, hand_motion_feat)

        vq_out["raw_features"] = {
            "body_shape": body_shape_feat,
            "hand_shape": hand_shape_feat,
            "body_motion": body_motion_feat,
            "hand_motion": hand_motion_feat,
        }
        return vq_out

    def get_tokens(self, pose_seq: torch.Tensor,
                   mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Return only the discrete token indices."""
        return self.encode(pose_seq, mask)["indices"]

    def embed_tokens(
        self,
        gs_idx: torch.Tensor,
        ls_idx: torch.Tensor,
        gm_idx: torch.Tensor,
        lm_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Embed 4 sets of indices and concatenate along feature dim.

        Returns: (B, T, 4*D).
        """
        embs = self.vq.embed_indices(gs_idx, ls_idx, gm_idx, lm_idx)
        return torch.cat([
            embs["global_shape"],
            embs["local_shape"],
            embs["global_motion"],
            embs["local_motion"],
        ], dim=-1)

    def quantized_to_decoder_input(self, vq_out: dict) -> torch.Tensor:
        """Concatenate quantized embeddings into decoder input (B, T, 4*D)."""
        q = vq_out["quantized"]
        return torch.cat([
            q["global_shape"],
            q["local_shape"],
            q["global_motion"],
            q["local_motion"],
        ], dim=-1)
