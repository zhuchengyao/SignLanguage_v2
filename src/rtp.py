"""RTP (Rhythm-Tempered Phase) module.

Predicts a per-frame continuous phase signal φ(t) ∈ (0, 1) from text features,
which modulates the decoder's positional encoding to inject natural
acceleration/deceleration rhythm into the generated motion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RTPModule(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        hidden_dim: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 500,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable temporal queries: will be expanded to target length
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("temporal_pe", pe.unsqueeze(0))

        # Cross-attention: temporal queries attend to text features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        target_len: int,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        text_features: (B, L_text, text_dim) — BERT hidden states.
        target_len: int — number of output frames T.
        text_mask: (B, L_text) bool — True for valid text tokens.

        Returns: phase (B, T, 1) in (0, 1).
        """
        B = text_features.size(0)

        memory = self.text_proj(text_features)

        queries = self.temporal_pe[:, :target_len].expand(B, -1, -1)
        queries = self.query_proj(queries)

        memory_key_padding_mask = ~text_mask if text_mask is not None else None

        h = self.cross_attn(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        phase = self.phase_head(h)  # (B, T, 1)
        return phase

    def modulate_pe(
        self,
        positional_encoding: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """Apply phase modulation to positional encoding.

        PE'(t) = PE(t) * (1 + alpha * (2*phase(t) - 1))
        This scales the PE between (1-alpha) and (1+alpha).
        """
        scale = 1.0 + self.alpha * (2.0 * phase - 1.0)
        return positional_encoding * scale


def compute_gt_rhythm(pose_seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute ground-truth rhythm signal from pose velocity norms.

    Returns normalised speed curve (B, T, 1) in [0, 1].
    """
    delta = pose_seq[:, 1:] - pose_seq[:, :-1]
    speed = delta.norm(dim=-1, keepdim=True)  # (B, T-1, 1)

    # Pad first frame with zero speed
    zero = torch.zeros(speed.size(0), 1, 1, device=speed.device, dtype=speed.dtype)
    speed = torch.cat([zero, speed], dim=1)  # (B, T, 1)

    if mask is not None:
        speed = speed * mask.unsqueeze(-1).float()

    # Normalise per-sequence to [0, 1]
    s_max = speed.amax(dim=1, keepdim=True).clamp(min=1e-6)
    rhythm = speed / s_max

    return rhythm
