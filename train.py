# ======================================================================
#  Training script for the **Spatio-Temporal Text-to-Pose Diffusion**
#  Compatible with:
#     • model.py          (TextToPoseDiffusion, 方案 B)
#     • config.py         (ModelConfig & TrainConfig)
#     • data_loader.py    (create_data_loaders)
# ======================================================================

from __future__ import annotations
import os, time, math, json, torch, numpy as np
import torch.optim as optim
from dataclasses import asdict
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import TextToPoseDiffusion                     # CHG
from config import ModelConfig, TrainConfig               # NEW
from types import SimpleNamespace                         # NEW

# ---------------- Project-internal modules ----------------
from data_loader import create_data_loaders

# ───────────────────────────────────────────────────────────
#  配置实例化
# ───────────────────────────────────────────────────────────
m_cfg = ModelConfig()   # 模型 & 数据相关
t_cfg = TrainConfig()   # 优化器 & 路径等

# 让 data_loader 继续只吃一个 cfg —— 打个简单 namespace
data_cfg = SimpleNamespace(**vars(m_cfg), **vars(t_cfg))  # NEW


# ======================================================================
# Learning-rate Scheduler  (warm-up + cosine)
# ======================================================================
def build_scheduler(optimizer: optim.Optimizer,
                    cfg: TrainConfig,             # CHG
                    steps_per_epoch: int):
    warm_steps  = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.num_epochs   * steps_per_epoch

    def lr_lambda(step):
        if step < warm_steps:
            return step / max(1, warm_steps)
        progress = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


# ======================================================================
# Checkpoint helpers
# ======================================================================
def save_checkpoint(model, optimizer, scheduler, scaler,
                    epoch: int, loss: float, path: str,
                    best_loss: float | None = None):
    ckpt = {
        "epoch": epoch,
        "loss": loss,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        # NEW ─ 同时保存两份 config，方便复现 / 推断
        "model_cfg": asdict(m_cfg),
        "train_cfg": asdict(t_cfg),
        "model_type": "text_to_pose_diffusion_v2",
    }
    torch.save(ckpt, path)
    print(f"💾  Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, path: str):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt.get("best_loss", float("inf"))


# ======================================================================
# Train / Eval loops
# ======================================================================
def train_epoch(model, loader, optimizer, scheduler, scaler, device, ep):
    model.train(); tot, n = 0.0, 0
    bar = tqdm(loader, desc=f"Epoch {ep}")
    for txt, pose in bar:
        pose = pose.to(device)
        with autocast(enabled=t_cfg.mixed_precision):        # CHG
            loss = model(txt, pose)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       t_cfg.gradient_clip_norm)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        tot += loss.item(); n += 1
        bar.set_postfix(loss=f"{loss.item():.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return tot / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); tot, n = 0.0, 0
    for txt, pose in loader:
        pose = pose.to(device)
        loss = model(txt, pose)
        tot += loss.item(); n += 1
    return tot / n


# ======================================================================
# Mean / Std pre-computation  (仅首次运行需要)
# ======================================================================
def compute_dataset_stats(loader):
    sums = torch.zeros(m_cfg.pose_dim)
    sqs  = torch.zeros(m_cfg.pose_dim)
    count = 0
    for _, pose in tqdm(loader, desc="📊 Computing mean/std"):
        pose = pose.view(-1, m_cfg.pose_dim)
        sums += pose.sum(0)
        sqs  += (pose ** 2).sum(0)
        count += pose.shape[0]
    mean = sums / count
    var  = sqs / count - mean ** 2
    std  = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean, std


# ======================================================================
# Main
# ======================================================================
def main():
    device = torch.device(m_cfg.device)            # CHG
    torch.manual_seed(42); np.random.seed(42)
    print("► Device:", device)

    # -------- Data loaders --------
    tr_loader, val_loader, test_loader = create_data_loaders(data_cfg)

    # -------- Dataset stats (once) --------
    if m_cfg.pose_normalize and torch.all(m_cfg.std == 1):
        mean, std = compute_dataset_stats(tr_loader)
        m_cfg.mean, m_cfg.std = mean, std
        tr_loader, val_loader, test_loader = create_data_loaders(data_cfg)
        print("✅  Pose mean / std computed & loaders rebuilt")

    # -------- Model & optimiser --------
    model = TextToPoseDiffusion(m_cfg).to(device)                        # CHG
    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(optim_params, lr=t_cfg.learning_rate, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, t_cfg, len(tr_loader))        # CHG
    scaler = GradScaler(enabled=t_cfg.mixed_precision)                   # CHG

    # -------- Logging / checkpoint dir --------
    os.makedirs(t_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(t_cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(t_cfg.log_dir)

    latest_ckpt = os.path.join(t_cfg.checkpoint_dir, "latest.pth")
    start_ep, best = 0, float("inf")
    if os.path.exists(latest_ckpt):
        print(f"🔄  Resuming from {latest_ckpt}")
        start_ep, best = load_checkpoint(model, optimizer, scheduler, scaler, latest_ckpt)
        start_ep += 1
        print(f"▶️  Start epoch {start_ep}, best val {best:.4f}")
    else:
        print("🆕  Fresh training run")

    no_up, patience = 0, 30
    for ep in range(start_ep, t_cfg.num_epochs):
        t0 = time.time()
        train_loss = train_epoch(model, tr_loader, optimizer, scheduler, scaler, device, ep)
        val_loss   = eval_epoch(model, val_loader, device)
        dt = time.time() - t0

        lr_now = scheduler.get_last_lr()[0]
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, ep)
        writer.add_scalar("LR", lr_now, ep)
        print(f"Epoch {ep:>4d} │ train {train_loss:.4f} │ val {val_loss:.4f} │ "
              f"lr {lr_now:.2e} │ {dt:.1f}s")

        # ---- checkpointing ----
        save_checkpoint(model, optimizer, scheduler, scaler, ep,
                        train_loss, latest_ckpt, best_loss=best)
        if ep % t_cfg.save_every == 0:
            periodic = os.path.join(t_cfg.checkpoint_dir, f"epoch_{ep}.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, ep,
                            train_loss, periodic, best_loss=best)
        if val_loss < best:
            best, no_up = val_loss, 0
            best_path = os.path.join(t_cfg.checkpoint_dir, "best.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, ep,
                            best, best_path, best_loss=best)
            print(f"🎯  New best! val {val_loss:.6f}")
        else:
            no_up += 1
            if no_up >= patience:
                print(f"⏹️  Early stop after {patience} epochs without improvement")
                break

    # -------- Testing --------
    best_path = os.path.join(t_cfg.checkpoint_dir, "best.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print("✅  Best model loaded for test")
    test_loss = eval_epoch(model, test_loader, device)
    print(f"📊  Final test loss: {test_loss:.6f}")

    # -------- Demo generation --------
    model.eval(); print("\n🎨  Sampling demo poses…")
    demo_texts = ["hello", "thank you", "water", "help", "good"]
    poses = model.sample(demo_texts, num_steps=20)    # (5, T, 150)
    print(f"Generated {poses.shape} | range[{poses.min():.2f},{poses.max():.2f}] "
          f"| mean {poses.mean():.2f} ± {poses.std():.2f}")

    writer.close()
    print("\n🏁  Training complete")


if __name__ == "__main__":
    main()
