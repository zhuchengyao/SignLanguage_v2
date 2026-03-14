"""Non-autoregressive decoder, token predictor, and length predictor.

NAR Decoder: maps embedded discrete latent tokens → continuous pose,
             conditioned on text features and RTP phase modulation.
Token Predictor: predicts HLC discrete tokens from text (for inference).
Length Predictor: predicts output sequence length from text CLS feature.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_hlc import HLC_NAR_Config


# ---------------------------------------------------------------------------
# Shared positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

    def get_pe(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


# ---------------------------------------------------------------------------
# Length Predictor
# ---------------------------------------------------------------------------

class LengthPredictor(nn.Module):
    """Predict output sequence length T from text CLS embedding."""

    def __init__(self, text_dim: int = 768, hidden: int = 256, max_len: int = 200):
        super().__init__()
        self.max_len = max_len
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, text_cls: torch.Tensor) -> torch.Tensor:
        """text_cls: (B, text_dim). Returns (B,) predicted length (continuous)."""
        raw = self.net(text_cls).squeeze(-1)
        return raw.clamp(1.0, float(self.max_len))

    @torch.no_grad()
    def predict(self, text_cls: torch.Tensor) -> torch.Tensor:
        """Returns (B,) integer lengths."""
        return self.forward(text_cls).round().long().clamp(min=1, max=self.max_len)


# ---------------------------------------------------------------------------
# Token Predictor
# ---------------------------------------------------------------------------

class TokenPredictor(nn.Module):
    """Predict HLC discrete tokens from text features (for inference).

    Uses cross-attention Transformer layers: learnable temporal queries
    attend to text features, then project to token logits for each codebook.
    """

    def __init__(self, cfg: HLC_NAR_Config):
        super().__init__()
        d = cfg.nar_d_model

        self.text_proj = nn.Linear(cfg.text_hidden_dim, d)
        self.pos_enc = PositionalEncoding(d, cfg.max_seq_len)

        self.query_embed = nn.Parameter(torch.randn(1, cfg.max_seq_len, d) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.tok_pred_nhead,
            dim_feedforward=cfg.tok_pred_dim_ff, dropout=cfg.tok_pred_dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.tok_pred_layers)
        self.norm = nn.LayerNorm(d)

        self.head_gs = nn.Linear(d, cfg.global_codebook_size)
        self.head_ls = nn.Linear(d, cfg.local_codebook_size)
        self.head_gm = nn.Linear(d, cfg.global_codebook_size)
        self.head_lm = nn.Linear(d, cfg.local_codebook_size)

    def forward(
        self,
        text_features: torch.Tensor,
        target_len: int,
        text_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        text_features: (B, L_text, text_dim).
        target_len: number of output time steps.
        Returns dict of logits for each codebook: (B, T, codebook_size).
        """
        B = text_features.size(0)
        memory = self.text_proj(text_features)

        queries = self.query_embed[:, :target_len].expand(B, -1, -1)
        queries = self.pos_enc(queries)

        mem_mask = ~text_mask if text_mask is not None else None

        h = self.decoder(tgt=queries, memory=memory,
                         memory_key_padding_mask=mem_mask)
        h = self.norm(h)

        return {
            "global_shape": self.head_gs(h),
            "local_shape": self.head_ls(h),
            "global_motion": self.head_gm(h),
            "local_motion": self.head_lm(h),
        }

    @torch.no_grad()
    def predict_tokens(
        self,
        text_features: torch.Tensor,
        target_len: int,
        text_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Return argmax token indices for each codebook."""
        logits = self.forward(text_features, target_len, text_mask)
        return {k: v.argmax(dim=-1) for k, v in logits.items()}


# ---------------------------------------------------------------------------
# NAR Decoder
# ---------------------------------------------------------------------------

class NARDecoder(nn.Module):
    """Non-autoregressive Transformer decoder.

    Takes concatenated quantized token embeddings (from HLC or Token Predictor),
    cross-attends to text features, and outputs continuous pose coordinates.
    RTP phase is applied to position encoding before decoding.
    """

    def __init__(self, cfg: HLC_NAR_Config):
        super().__init__()
        d = cfg.nar_d_model
        self.d_model = d

        self.input_proj = nn.Linear(cfg.nar_input_dim, d)
        self.pos_enc = PositionalEncoding(d, cfg.max_seq_len)

        self.text_proj = nn.Linear(cfg.text_hidden_dim, d)

        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.nar_nhead,
            dim_feedforward=cfg.nar_dim_ff, dropout=cfg.nar_dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=cfg.nar_num_layers)
        self.norm = nn.LayerNorm(d)

        # Output: 50 joints * 2 coordinates = 100
        self.output_head = nn.Linear(d, cfg.num_joints * cfg.coord_dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        text_features: torch.Tensor,
        phase: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        rtp_module=None,
    ) -> torch.Tensor:
        """
        token_embeddings: (B, T, nar_input_dim) from HLC or token predictor.
        text_features: (B, L_text, text_dim).
        phase: (B, T, 1) from RTP module.
        text_mask: (B, L_text) bool.
        target_mask: (B, T) bool.

        Returns: (B, T, 100) predicted xy coordinates for 50 joints.
        """
        h = self.input_proj(token_embeddings)

        pe = self.pos_enc.get_pe(h.size(1)).expand(h.size(0), -1, -1)
        if phase is not None and rtp_module is not None:
            pe = rtp_module.modulate_pe(pe, phase)
        h = h + pe

        memory = self.text_proj(text_features)

        mem_mask = ~text_mask if text_mask is not None else None
        tgt_mask = ~target_mask if target_mask is not None else None

        decoded = self.transformer(
            tgt=h,
            memory=memory,
            memory_key_padding_mask=mem_mask,
            tgt_key_padding_mask=tgt_mask,
        )
        decoded = self.norm(decoded)

        return self.output_head(decoded)
