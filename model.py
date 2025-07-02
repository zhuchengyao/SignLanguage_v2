# model.py
from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

try:
    import torch_geometric.nn as gnn
except ImportError:
    gnn = None

from skeletalModel import getSkeletalModelStructure
from config import ModelConfig


# ════════════════════════════════════════════════════════════════
# Helper: build undirected edge_index (2, E) for 50-node skeleton
# ════════════════════════════════════════════════════════════════
def build_edge_index(num_joints: int = 50) -> torch.Tensor:
    triplets: List[Tuple[int, int, str]] = list(getSkeletalModelStructure())
    edges = {(s, e) for s, e, _ in triplets} | {(e, s) for s, e, _ in triplets}
    return torch.tensor(sorted(edges), dtype=torch.long).t()   # (2, E)


# ════════════════════════
# Text Encoder  (BERT-CLS)
# ════════════════════════
class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(dtype).eval()
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.proj = nn.Linear(768, cfg.text_embed_dim, bias=True).to(dtype)

    @torch.no_grad()
    def forward(self, texts: List[str] | str) -> torch.Tensor:          # (B, D_txt)
        if isinstance(texts, str):
            texts = [texts]
        toks = self.tokenizer(texts, return_tensors="pt", padding=True,
                              truncation=True, max_length=32)
        toks = {k: v.to(self.proj.weight.device) for k, v in toks.items()}
        cls = self.bert(**toks).last_hidden_state[:, 0]
        cls = cls.to(self.proj.weight.dtype)
        return self.proj(cls)


# ════════════════════════
# Pose Encoder  (GAT 或 MLP)
# ════════════════════════
class PoseGAT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.num_joints = 50
        self.use_gat = gnn is not None

        if self.use_gat:
            dims = (3, *cfg.gat_hidden_dims)
            heads = list(cfg.gat_heads or [4] * (len(dims) - 1))

            self.register_buffer("edge_index", build_edge_index(self.num_joints))
            self.gconvs = nn.ModuleList([
                gnn.GATConv(
                    in_channels=dims[i],
                    out_channels=dims[i + 1] // heads[i],  # concat=True
                    heads=heads[i],
                    concat=True,
                ) for i in range(len(dims) - 1)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])
            self.res_proj = nn.ModuleList([
                nn.Identity() if dims[i] == dims[i + 1]
                else nn.Linear(dims[i], dims[i + 1], bias=False)
                for i in range(len(dims) - 1)
            ])
            self.act = nn.GELU()
            self.frame_proj = nn.Linear(self.num_joints * dims[-1],
                                        cfg.pose_embed_dim)
        else:  # 简化版：MLP
            self.mlp = nn.Sequential(
                nn.Linear(self.num_joints * 3, 512),
                nn.GELU(),
                nn.Linear(512, cfg.pose_embed_dim),
            )

        self.final_norm = nn.LayerNorm(cfg.pose_embed_dim)

    def forward(self, pose_seq: torch.Tensor) -> torch.Tensor:
        """
        pose_seq: (B, T, J, 3) or (B, T, J*3)
        return : (B, T, D_pose)
        """
        if pose_seq.dim() == 4:
            B, T, J, _ = pose_seq.shape
            pose_seq = pose_seq.view(B, T, J * 3)
        B, T, _ = pose_seq.shape
        assert T == self.cfg.clip_len, f"expected clip_len={self.cfg.clip_len}, got {T}"

        if self.use_gat:
            x = pose_seq.view(B * T, self.num_joints, 3)          # (B*T, J, 3)
            device = x.device
            edge_base = self.edge_index.to(device)
            E = edge_base.size(1)
            batch_offset = (torch.arange(B * T, device=device)
                            .repeat_interleave(E).view(1, -1)) * self.num_joints
            edge = edge_base.repeat(1, B * T) + batch_offset      # (2, E*B*T)
            x = x.view(-1, 3)                                     # (B*T*J, 3)

            for i, (conv, ln) in enumerate(zip(self.gconvs, self.norms)):
                h = conv(x, edge)
                res = x if isinstance(self.res_proj[i], nn.Identity) \
                       else self.res_proj[i](x)
                x = self.act(ln(h + res))
            x = x.view(B * T, self.num_joints, -1).reshape(B * T, -1)
            x = self.frame_proj(x)                                # (B*T, D_pose)
        else:
            x = self.mlp(pose_seq.view(B * T, -1))

        x = self.final_norm(x).view(B, T, -1)                     # (B, T, D_pose)
        return x


# ════════════════════════
# 可学习 Sin/Cos 时间编码
# ════════════════════════
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_time: int):
        super().__init__()
        self.dim = dim
        self.max_time = max_time
        self.linear = nn.Linear(dim, dim)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, t: torch.Tensor) -> torch.Tensor:           # t: (B,) or (B, T)
        max_norm = max(1, self.max_time - 1)
        half = self.dim // 2
        freqs = torch.exp(torch.arange(
            half, device=t.device, dtype=torch.float32
        ) * (-math.log(10000.0) / (half - 1)))
        phase = (t.to(torch.float32)[..., None] / max_norm) * freqs
        pe = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return self.linear(pe)


# ════════════════════════
# Transformer-based Decoder
# ════════════════════════
class PoseTransformerDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_proj = nn.Linear(cfg.pose_embed_dim, cfg.hidden_dim)
        self.time_embed = TimeEmbedding(cfg.hidden_dim, cfg.clip_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.decoder_heads,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ) for _ in range(cfg.decoder_layers)
        ])
        self.out = nn.Linear(cfg.hidden_dim, cfg.pose_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:      # (B, T, D_pose)
        dtype = self.latent_proj.weight.dtype
        latent = latent.to(dtype)
        B, T, _ = latent.shape

        x = self.latent_proj(latent)                              # (B, T, H)
        t_idx = torch.arange(T, device=latent.device)[None, :].expand(B, -1)
        x = x + self.time_embed(t_idx)

        for layer in self.layers:
            x = layer(x)
        out = self.out(x)                                         # (B, T, J*3)

        if self.cfg.pose_normalize:
            mean = self.cfg.mean.to(out.device, dtype=out.dtype)
            std = self.cfg.std.to(out.device, dtype=out.dtype)
            out = out * std + mean
        return out


# ════════════════════════
#  空-时 Transformer Block
# ════════════════════════
class STBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, H)
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        return self.norm2(x + self.ff(x))


# ════════════════════════
#  Diffusion ε-Predictor
# ════════════════════════
class NoisePredictor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.t_embed = TimeEmbedding(cfg.hidden_dim, cfg.num_diffusion_steps)
        self.pose_proj = nn.Linear(cfg.pose_embed_dim, cfg.hidden_dim)
        self.text_proj = nn.Linear(cfg.text_embed_dim, cfg.hidden_dim)
        self.gamma = nn.Parameter(torch.ones(1))
        self.blocks = nn.ModuleList([STBlock(cfg.hidden_dim, 8)
                                     for _ in range(cfg.num_layers)])
        self.out = nn.Linear(cfg.hidden_dim, cfg.pose_embed_dim)

    def forward(self,
                pose_e: torch.Tensor,        # (B, T, D_pose)
                text_e: torch.Tensor,        # (B, D_text)
                t: torch.Tensor) -> torch.Tensor:
        dtype = self.pose_proj.weight.dtype
        pose_e = pose_e.to(dtype)
        text_e = text_e.to(dtype)

        x = self.pose_proj(pose_e)                              # (B, T, H)
        x = x + self.text_proj(text_e).unsqueeze(1)             # broadcast
        x = x + self.gamma * self.t_embed(t).unsqueeze(1)       # 时间嵌入
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)                                      # (B, T, D_pose)


# ════════════════════════
#  Diffusion Scheduler
# ════════════════════════
class DiffusionScheduler(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        betas = torch.linspace(cfg.beta_start, cfg.beta_end,
                               cfg.num_diffusion_steps,
                               dtype=torch.float32)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_1m_alphas_cumprod",
                             torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod",
                             torch.sqrt(self.alphas_cumprod))

    def add_noise(self,
                  x0: torch.Tensor,          # (B, T, D)
                  noise: torch.Tensor,       # (B, T, D)
                  t: torch.Tensor            # (B,)
                  ) -> torch.Tensor:
        a_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sigma = self.sqrt_1m_alphas_cumprod[t].view(-1, 1, 1)
        return a_bar * x0 + sigma * noise


# ════════════════════════
#  Full Text→Pose Diffusion
# ════════════════════════
class TextToPoseDiffusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.txt_e = TextEncoder(cfg)
        self.pose_e = PoseGAT(cfg)
        self.noise = NoisePredictor(cfg)
        self.dec = PoseTransformerDecoder(cfg)
        self.sched = DiffusionScheduler(cfg)
        print("🚀  Text-to-Pose Diffusion (spatio-temporal) ready!")

    # ─────────────────────────────────────────────
    # Train-time forward   ε-loss + recon-loss
    # ─────────────────────────────────────────────
    def forward(self, texts: List[str] | str,
                pose_seq: torch.Tensor) -> torch.Tensor:
        """
        pose_seq: (B, T, J, 3)  or (B, T, J*3)
        """
        B, device = pose_seq.size(0), pose_seq.device
        text_e = self.txt_e(texts)
        pose_e = self.pose_e(pose_seq)                         # (B, T, D)

        t = torch.randint(0, self.cfg.num_diffusion_steps,
                          (B,), device=device)
        noise = torch.randn_like(pose_e)
        noisy = self.sched.add_noise(pose_e, noise, t)
        pred_noise = self.noise(noisy, text_e, t)
        l_noise = nn.functional.mse_loss(pred_noise, noise)

        recon = self.dec(pose_e)                               # (B, T, J*3)
        l_recon = nn.functional.mse_loss(
            recon, pose_seq.view(B, self.cfg.clip_len, -1))
        return l_noise + self.cfg.recon_loss_weight * l_recon

    # ─────────────────────────────────────────────
    # DDIM-like deterministic sampling
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, texts: List[str] | str,
               num_steps: int = 50) -> torch.Tensor:           # (B, T, J*3)
        device = next(self.parameters()).device
        B = len(texts) if isinstance(texts, (list, tuple)) else 1
        text_e = self.txt_e(texts).to(device)

        x = torch.randn(B, self.cfg.clip_len,
                        self.cfg.pose_embed_dim, device=device)
        ts = torch.linspace(self.cfg.num_diffusion_steps - 1, 0,
                            num_steps, device=device).round().long()
        ts = torch.unique_consecutive(ts)
        if ts[-1] != 0:
            ts = torch.cat([ts, torch.zeros(1, dtype=torch.long,
                                            device=device)])

        for i, t in enumerate(ts):
            t_batch = t.expand(B)
            eps = self.noise(x, text_e, t_batch)
            alpha_bar_t = self.sched.alphas_cumprod[t]
            sqrt_alpha_bar = alpha_bar_t.sqrt()
            sqrt_1m_alpha_bar = (1 - alpha_bar_t).sqrt()
            x0_pred = (x - sqrt_1m_alpha_bar * eps) / sqrt_alpha_bar

            if t == 0:
                x = x0_pred
                break

            t_prev = ts[i + 1]
            alpha_bar_prev = self.sched.alphas_cumprod[t_prev]
            sqrt_alpha_prev = alpha_bar_prev.sqrt()
            sqrt_1m_alpha_prev = (1 - alpha_bar_prev).sqrt()
            x = sqrt_alpha_prev * x0_pred + sqrt_1m_alpha_prev * eps

        return self.dec(x)                                     # (B, T, J*3)
