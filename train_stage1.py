"""
stage1_train_final.py
─────────────────────
• 大网络  (pose_embed=1024, hidden=2048)
• lr=3e-3, recon_loss_weight=100
• 不再 clamp
目标：10-15 epoch 内  val Recon MSE ≤ 0.01
"""
from pathlib import Path
import time, torch, numpy as np
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from model   import TextToPoseDiffusion
from config  import TrainConfig
from data_loader_fixed import create_data_loaders, ASLPoseDataset

cfg = TrainConfig()
cfg.pose_normalize = False
cfg.batch_size     = 32
cfg.pose_embed_dim = 1024
cfg.hidden_dim     = 2048
cfg.num_epochs     = 25
cfg.learning_rate  = 3e-3           # ← 大步
cfg.recon_loss_weight = 100.0
cfg.pose_clip_range = 1e9           # ← 等同 “不 clamp”

run = Path("runs_final_"+time.strftime("%m%d_%H%M%S")); run.mkdir()
writer = SummaryWriter(str(run))

tr, dv, _ = create_data_loaders(cfg)

# 逐维 μ⃗/σ⃗
all_frames = np.array([f for seq in tr.dataset.poses for f in seq])
mu  = torch.tensor(all_frames.mean(0), dtype=torch.float32, device=cfg.device)
std = torch.tensor(all_frames.std (0), dtype=torch.float32, device=cfg.device)
std[std < 1e-6] = 1.0

print("μ⃗/σ⃗ ready  σ̄ =", std.mean().item())

model = TextToPoseDiffusion(cfg).to(cfg.device)
for p in model.noise.parameters(): p.requires_grad_(False)
opt  = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)

step=0
for ep in range(cfg.num_epochs):
    model.train()
    for _, pose in tr:
        pose = pose.to(cfg.device)                          # (B,T,150)
        norm = (pose - mu) / std                            # no clamp
        norm = norm.view(norm.size(0), norm.size(1), 50, 3)

        recon = model.dec(model.pose_e(norm))
        loss  = torch.nn.functional.mse_loss(recon, norm.view_as(recon)) * cfg.recon_loss_weight
        opt.zero_grad(); loss.backward(); clip_grad_norm_(model.parameters(),1); opt.step()

        if step % 50 == 0: writer.add_scalar("train/recon_mse", loss.item(), step)
        step += 1

    # —— validation ——
    model.eval(); err=0.; tot=0
    with torch.no_grad():
        for _, pose in dv:
            pose = pose.to(cfg.device)
            norm = ((pose - mu) / std).view(pose.size(0), pose.size(1), 50, 3)
            rec  = model.dec(model.pose_e(norm))
            err += torch.nn.functional.mse_loss(rec, norm.view_as(rec), reduction="sum").item()
            tot += norm.numel()
    mse = err / tot
    writer.add_scalar("val/recon_mse", mse, ep)
    print(f"[Ep {ep:02d}] val Recon MSE = {mse:.6f}")

print("✅ Stage-1 (final) 结束 – 目标 ≤ 0.01；达标即可 Stage-2")
