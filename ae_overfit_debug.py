"""
ae_overfit_debug.py
───────────────────
• 只取 1 条 clip 反复训练 300 步
• 防止 σ = 0 → NaN：std<1e-6 时替换为 1
• 网络用与主训练相同的 cfg，但可临时放大 hidden_dim 便于收敛
"""

import torch, numpy as np
from data_loader_fixed import ASLPoseDataset
from config import TrainConfig
from model import TextToPoseDiffusion

# ────────── 1. 读取一条样本 ──────────
cfg = TrainConfig(); cfg.pose_normalize = False
dataset = ASLPoseDataset(cfg.data_root, "train", clip_len=cfg.clip_len)
_, pose = dataset[0]                      # pose: (T,150)
pose = pose.unsqueeze(0).to(cfg.device)   # (1,T,150)

# ────────── 2. 逐维 μ⃗/σ⃗ + 防除 0 ──────────
mu  = pose.view(-1, 150).mean(0)
std = pose.view(-1, 150).std (0)
std_safe = std.clone()
std_safe[std_safe < 1e-6] = 1.0           # ★关键：σ=0 → 1

norm = ((pose - mu) / std_safe).clamp(-8, 8)
norm = norm.view(1, cfg.clip_len, 50, 3)  # (1,T,50,3)

# ────────── 3. 构建模型（可放宽容量）──────────
cfg.pose_embed_dim = 512
cfg.hidden_dim     = 1024
model = TextToPoseDiffusion(cfg).to(cfg.device)
for p in model.noise.parameters():        # 只训 Encoder/Decoder
    p.requires_grad_(False)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# ────────── 4. 训练 300 步 ──────────
for step in range(301):
    pred = model.dec(model.pose_e(norm))
    loss = torch.nn.functional.mse_loss(pred, norm.view_as(pred))
    optim.zero_grad(); loss.backward(); optim.step()
    if step % 50 == 0:
        print(f"step {step:03d}  loss {loss.item():.6f}")
