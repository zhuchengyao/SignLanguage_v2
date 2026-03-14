"""Unified HLC-NAR model combining all components.

Training mode: GT pose → HLC → discrete tokens → NAR Decoder → reconstructed pose
Inference mode: text → Token Predictor → tokens → NAR Decoder → pose
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .config_hlc import HLC_NAR_Config
from .hlc.hlc_module import HLCModule
from .model_nar import NARDecoder, TokenPredictor, LengthPredictor
from .rtp import RTPModule, compute_gt_rhythm
from .losses_kals import kals_loss


class HLC_NAR_Model(nn.Module):
    """End-to-end Text→Pose sign language generation model."""

    def __init__(self, cfg: HLC_NAR_Config):
        super().__init__()
        self.cfg = cfg

        # 1. Text encoder (frozen BERT)
        self.text_encoder = BertModel.from_pretrained(cfg.text_model_name)
        if cfg.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # 2. HLC module (shape + motion encoders + hierarchical VQ)
        self.hlc = HLCModule(cfg)

        # 3. Non-autoregressive decoder
        self.nar_decoder = NARDecoder(cfg)

        # 4. Token predictor (text → discrete tokens, for inference)
        self.token_predictor = TokenPredictor(cfg)

        # 5. Length predictor
        self.length_predictor = LengthPredictor(
            text_dim=cfg.text_hidden_dim,
            hidden=cfg.len_pred_hidden,
            max_len=cfg.max_target_len,
        )

        # 6. RTP module
        self.rtp = RTPModule(
            text_dim=cfg.text_hidden_dim,
            hidden_dim=cfg.rtp_hidden_dim,
            nhead=cfg.rtp_nhead,
            num_layers=cfg.rtp_num_layers,
            max_len=cfg.max_seq_len,
            alpha=cfg.rtp_alpha,
        )

        # Cache for mean/std (set externally by training script)
        self.register_buffer("pose_mean", torch.zeros(150), persistent=False)
        self.register_buffer("pose_std", torch.ones(150), persistent=False)

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(
        self, tokenized_text: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_features: (B, L, 768) full hidden states
            text_cls: (B, 768) CLS token
            text_mask: (B, L) bool
        """
        with torch.no_grad() if self.cfg.freeze_text_encoder else torch.enable_grad():
            out = self.text_encoder(**tokenized_text)
        text_features = out.last_hidden_state
        text_cls = text_features[:, 0]
        text_mask = tokenized_text["attention_mask"].bool()
        return text_features, text_cls, text_mask

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_train(
        self,
        tokenized_text: dict,
        pose_seq: torch.Tensor,
        pose_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Full training forward pass.
        pose_seq: (B, T, 150) normalised.
        pose_mask: (B, T) bool.
        Returns dict with all losses and intermediates.
        """
        B, T, _ = pose_seq.shape

        # --- Text ---
        text_features, text_cls, text_mask = self.encode_text(tokenized_text)

        # --- HLC encode GT pose ---
        hlc_out = self.hlc.encode(pose_seq, pose_mask)
        decoder_input = self.hlc.quantized_to_decoder_input(hlc_out)  # (B, T, 4*D)

        # --- Length prediction ---
        pred_len = self.length_predictor(text_cls)
        gt_len = pose_mask.sum(dim=1).float()

        # --- RTP ---
        phase = self.rtp(text_features, T, text_mask)
        gt_rhythm = compute_gt_rhythm(pose_seq, pose_mask)

        # --- NAR Decode from HLC tokens ---
        pred_xy_flat = self.nar_decoder(
            decoder_input, text_features,
            phase=phase, text_mask=text_mask, target_mask=pose_mask,
            rtp_module=self.rtp,
        )  # (B, T, 100)

        # --- Reconstruction loss ---
        pred_xy = pred_xy_flat.view(B, T, 50, 2)

        gt_body = pose_seq[..., :24].reshape(B, T, 8, 3)[..., :2]
        gt_right = pose_seq[..., 24:87].reshape(B, T, 21, 3)[..., :2]
        gt_left = pose_seq[..., 87:150].reshape(B, T, 21, 3)[..., :2]
        gt_xy = torch.cat([gt_body, gt_left, gt_right], dim=2)  # (B, T, 50, 2)

        valid = pose_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, T, 1, 1)

        body_recon = F.mse_loss(
            pred_xy[:, :, :8] * valid, gt_xy[:, :, :8] * valid, reduction="sum"
        )
        hand_recon = F.mse_loss(
            pred_xy[:, :, 8:] * valid, gt_xy[:, :, 8:] * valid, reduction="sum"
        )
        denom = valid.sum().clamp(min=1.0) * 50 * 2
        recon_loss = (
            self.cfg.body_recon_weight * body_recon
            + self.cfg.hand_recon_weight * hand_recon
        ) / denom

        # --- VQ loss ---
        vq_loss = hlc_out["vq_loss"]

        # --- Token prediction loss ---
        tok_logits = self.token_predictor(text_features, T, text_mask)
        gt_indices = hlc_out["indices"]
        tok_pred_loss = sum(
            F.cross_entropy(
                tok_logits[k].reshape(-1, tok_logits[k].size(-1)),
                gt_indices[k].reshape(-1),
                reduction="mean",
            )
            for k in gt_indices
        ) / len(gt_indices)

        # --- Length prediction loss ---
        len_loss = F.l1_loss(pred_len, gt_len)

        # --- RTP rhythm loss ---
        rtp_loss = F.mse_loss(phase * valid[..., 0], gt_rhythm * valid[..., 0])

        # --- KALS ---
        kals_total, kals_detail = kals_loss(
            pred_xy, pose_mask,
            w_bone=self.cfg.kals_bone_weight,
            w_angle=self.cfg.kals_angle_weight,
            w_sym=self.cfg.kals_symmetry_weight,
        )

        return {
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "token_pred_loss": tok_pred_loss,
            "length_loss": len_loss,
            "rtp_loss": rtp_loss,
            "kals_loss": kals_total,
            "pred_xy": pred_xy.detach(),
            "phase": phase.detach(),
            **{f"kals_{k}": v for k, v in kals_detail.items()},
        }

    # ------------------------------------------------------------------
    # Inference forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_inference(
        self,
        tokenized_text: dict,
        target_len: int | None = None,
    ) -> Dict[str, Any]:
        """
        Single forward pass for inference.
        If target_len is None, uses the length predictor.
        Returns predicted pose_xy (B, T, 50, 2) and other info.
        """
        text_features, text_cls, text_mask = self.encode_text(tokenized_text)

        if target_len is None:
            T = self.length_predictor.predict(text_cls).item()
            T = max(1, min(int(T), self.cfg.max_target_len))
        else:
            T = target_len

        # Predict tokens from text
        tok_indices = self.token_predictor.predict_tokens(text_features, T, text_mask)

        # Embed tokens
        token_emb = self.hlc.embed_tokens(
            tok_indices["global_shape"],
            tok_indices["local_shape"],
            tok_indices["global_motion"],
            tok_indices["local_motion"],
        )

        # RTP phase
        phase = self.rtp(text_features, T, text_mask)

        # Decode
        pred_xy_flat = self.nar_decoder(
            token_emb, text_features,
            phase=phase, text_mask=text_mask,
            rtp_module=self.rtp,
        )
        pred_xy = pred_xy_flat.view(1, T, 50, 2)

        return {
            "pred_xy": pred_xy,
            "pred_len": T,
            "phase": phase,
            "token_indices": tok_indices,
        }

    # ------------------------------------------------------------------
    # Helpers for staged training
    # ------------------------------------------------------------------

    def freeze_hlc_codebooks(self):
        """Freeze VQ codebook embeddings (used in stage 2+)."""
        for vq in [
            self.hlc.vq.global_shape_vq,
            self.hlc.vq.local_shape_vq,
            self.hlc.vq.global_motion_vq,
            self.hlc.vq.local_motion_vq,
        ]:
            vq.embedding.weight.requires_grad_(False)

    def freeze_text_encoder(self):
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def get_stage1_params(self):
        """Parameters for stage 1: HLC + NAR Decoder."""
        params = list(self.hlc.parameters()) + list(self.nar_decoder.parameters())
        return params

    def get_stage2_params(self):
        """Parameters for stage 2: RTP + Token Predictor + NAR Decoder + Length Predictor."""
        params = (
            list(self.rtp.parameters())
            + list(self.token_predictor.parameters())
            + list(self.nar_decoder.parameters())
            + list(self.length_predictor.parameters())
        )
        return params

    def get_stage3_params(self):
        """Parameters for stage 3: everything except frozen text encoder."""
        params = [p for p in self.parameters() if p.requires_grad]
        return params
