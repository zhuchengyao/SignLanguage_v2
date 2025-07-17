# -------------------------------------------------
# modules.py (optimised)
# -------------------------------------------------
from __future__ import annotations
import math
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_geometric.nn as gnn
except ImportError:
    gnn = None  # Only required for GAT encoder


# ---------- generic helpers ----------

def get_skeletal_model_structure():
    BODY, HAND = 8, 21
    pose_conn = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]
    hand_conn = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    triplets = [(s, e, "body") for s, e in pose_conn]
    triplets += [(s + BODY, e + BODY, "lh") for s, e in hand_conn]
    triplets += [
        (s + BODY + HAND, e + BODY + HAND, "rh") for s, e in hand_conn
    ]
    return triplets


def build_edge_index(num_joints: int = 50):
    triplets = get_skeletal_model_structure()
    edges = {(s, e) for s, e, _ in triplets} | {(e, s) for s, e, _ in triplets}
    return torch.tensor(sorted(edges), dtype=torch.long).t()  # (2, E)


# ---------- Rotary Pos‑Embedding ----------
class RotaryEmbedding(nn.Module):
    """Pre‑computes θ for RoPE; caches on device for speed."""

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )  # (D/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)  # (T,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T,D/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T,D)
        return emb  # angles (rad)


def _rotate_half(x: torch.Tensor):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, angles: torch.Tensor):
    """Applies RoPE — expects q,k:(B,H,T,Dh)  angles:(T,Dh)."""
    cos, sin = angles.cos()[None, None, ...], angles.sin()[None, None, ...]
    q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
    return q, k


# ---------- Core Blocks ----------
class GatedMLP(nn.Module):
    """Temporal Gated‑MLP used at the tail of each DiT block."""

    def __init__(self, dim: int, expansion: int | None = None):
        super().__init__()
        hidden = expansion or dim * 2
        self.fc = nn.Linear(dim, hidden * 2)  # val ‖ gate
        self.out = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):  # (B,T,D)
        v, g = self.fc(x).chunk(2, dim=-1)
        return self.out(self.act(v) * torch.tanh(g))


class DiTBlock(nn.Module):
    """
    DiT transformer block with AdaLN‑Zero, optional RoPE & temporal‑gate.
    This variant eliminates **all** Python‑side loops & uses PyTorch 2.2
    fused kernels (scaled_dot_product_attention, FSDP‑friendly).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        use_rope: bool = True,
        use_temporal_gate: bool = True,
    ) -> None:
        super().__init__()
        self.dim, self.heads = dim, heads
        self.use_rope, self.use_temporal_gate = use_rope, use_temporal_gate

        # ---- layers ----
        self.norm_msa = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

        # AdaLN‑Zero (FiLM: shift/scale & residual gate for msa + mlp)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # RoPE pre‑compute (per‑head dim)
        if use_rope:
            self.rope = RotaryEmbedding(dim // heads)

        if use_temporal_gate:
            self.temporal_gate = GatedMLP(dim)

    # ------------------------
    def _rope_qkv(self, x: torch.Tensor):  # helper for RoPE path
        B, T, _ = x.shape
        q, k, v = self.attn.in_proj_weight.chunk(3)
        bq, bk, bv = self.attn.in_proj_bias.chunk(3)
        H = self.heads
        Dh = self.dim // H
        # shape -> (B,H,T,Dh)
        q = (x @ q.t() + bq).view(B, T, H, Dh).transpose(1, 2)
        k = (x @ k.t() + bk).view(B, T, H, Dh).transpose(1, 2)
        v = (x @ v.t() + bv).view(B, T, H, Dh).transpose(1, 2)
        # apply RoPE
        angles = self.rope(T, x.device)  # (T,Dh)
        q, k = apply_rope(q, k, angles)
        # attention
        y = F.scaled_dot_product_attention(q, k, v)  # (B,H,T,Dh)
        y = y.transpose(1, 2).reshape(B, T, self.dim)
        return self.attn.out_proj(y)

    # ------------------------
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # cond may be (B,D) or (B,L,D)
        if cond.dim() == 3:
            cond = cond.mean(dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.film(cond).unsqueeze(1).chunk(6, dim=-1)
        )

        # ---- MSA ----
        h = self.norm_msa(x) * (1 + scale_msa) + shift_msa
        attn_out = (
            self._rope_qkv(h) if self.use_rope else self.attn(h, h, h)[0]
        )
        x = x + gate_msa * attn_out

        # ---- MLP ----
        h = self.norm_mlp(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)

        # ---- temporal gated MLP ----
        if self.use_temporal_gate:
            x = x + self.temporal_gate(x)
        return x


# ---------- VAE enc/dec (GAT) ----------
class PoseGatEncoder(nn.Module):
    def __init__(self, num_joints, pose_embed_dim, gat_hidden_dims, gat_heads):
        super().__init__()
        if gnn is None:
            raise ImportError("torch_geometric must be installed to use PoseGatEncoder.")
        self.num_joints = num_joints

        dims, heads = (3, *gat_hidden_dims), list(gat_heads)
        self.register_buffer("edge_index", build_edge_index(num_joints))
        
        # ✅ 统一用老名字 convs，额外保留 gconvs 作为别名
        self.convs = nn.ModuleList([
            gnn.GATv2Conv(dims[i], dims[i+1] // heads[i],
                          heads=heads[i], concat=True)
            for i in range(len(dims) - 1)
        ])
        self.gconvs = self.convs      # 兼容新代码里的 gconvs

        self.gat_output_dim = self.num_joints * dims[-1]

        # ✅ 统一用老名字 out，保留 out_proj 兼容新调用
        self.out = nn.Linear(self.gat_output_dim, pose_embed_dim)
        self.out_proj = self.out

        self.act = nn.ELU()

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        device = x_seq.device
        num_edges, num_graphs = self.edge_index.size(1), B * T

        edge_index_batched = self.edge_index.repeat(1, num_graphs)
        offset = torch.arange(num_graphs, device=device).repeat_interleave(num_edges) * self.num_joints
        edge_index_batched += offset.view(1, -1)

        x = x_seq.view(B * T * self.num_joints, 3)
        for conv in self.convs:       # ← 这里也换成 convs
            x = self.act(conv(x, edge_index_batched))

        x = self.out(x.view(B * T, self.gat_output_dim))
        return x.view(B, T, -1)


class SpatialMlpDecoder(nn.Module):
    def __init__(self, pose_embed_dim, pose_dim, hidden_dim=512):
        super().__init__()
        # ✅ 老名字 mlp，保留 net 兼容
        self.mlp = nn.Sequential(
            nn.Linear(pose_embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pose_dim)
        )
        self.net = self.mlp

    def forward(self, pose_embed_seq: torch.Tensor) -> torch.Tensor:
        return self.mlp(pose_embed_seq)

# # --- ③ ★可选★ 加一道兜底：加载 VAE 时宽松匹配 -------------------------
# # motion_dit.py 里大约 55 行
# vae_state = torch.load(final_vae_path, map_location=self.device)
# self.vae.load_state_dict(vae_state, strict=False)   # ← strict=False，忽略多余键


class TimeEmbedding(nn.Module):
    """Classic 1‑D Sin/Cos ∈ ℝ^{B×D}."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):  # (B,)
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half - 1))
        )
        phase = t.float().unsqueeze(-1) * freqs  # (B,half)
        return torch.cat([phase.sin(), phase.cos()], dim=-1)  # (B,D)


class PositionalEncoding(nn.Module):
    """Absolute sin/cos PE kept for legacy modules (e.g., motion_vae.py)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):  # (B,T,D)
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class EMA:
    def __init__(self, params, decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.clone().detach() for p in params if p.requires_grad]

    @torch.no_grad()
    def update(self, params):
        for s, p in zip(self.shadow, [p for p in params if p.requires_grad]):
            s.lerp_(p.detach(), 1 - self.decay)

    def state_dict(self):
        return [p.clone() for p in self.shadow]