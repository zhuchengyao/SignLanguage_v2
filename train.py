# train.py (Stage 2: Latent Diffusion Training)
# ======================================================================
#  Training script · Latent Diffusion for Text-to-Pose
#  Compatible with:
#     • model.py          (LatentDiffusion)
#     • config.py         (ModelConfig for Stage 2)
#     • data_loader.py    (create_data_loaders → no changes needed)
# ======================================================================

from __future__ import annotations
import os, time, math, json, torch, numpy as np
import torch.optim as optim
from dataclasses import asdict
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------- project modules ----------
from model   import LatentDiffusion  # ✨ CHANGED: 导入新的 LatentDiffusion 模型
from config  import ModelConfig, TrainConfig
from data_loader import create_data_loaders
from types import SimpleNamespace

# ════════════════════════════════════════════════════════════════════
#  cfg objects
# ════════════════════════════════════════════════════════════════════
m_cfg = ModelConfig()
t_cfg = TrainConfig()

# 让 data_loader 只吃一个 cfg
data_cfg = SimpleNamespace(**vars(m_cfg), **vars(t_cfg))

# ════════════════════════════════════════════════════════════════════
#  LR scheduler (warm-up + cosine)
# (无需改动)
# ════════════════════════════════════════════════════════════════════
def build_scheduler(optimizer: optim.Optimizer, cfg: TrainConfig,
                    steps_per_epoch: int):
    warm_steps  = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.num_epochs   * steps_per_epoch

    def lr_lambda(step):
        if step < warm_steps:
            return step / max(1, warm_steps)
        progress = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

# ════════════════════════════════════════════════════════════════════
#  checkpoint utils
# ════════════════════════════════════════════════════════════════════
def save_checkpoint(model, optimizer, scheduler, scaler,
                    epoch: int, loss: float, path: str,
                    best_loss: float | None = None):
    ckpt = dict(
        epoch    = epoch,
        loss     = loss,
        best_loss= best_loss,
        model_state_dict     = model.state_dict(),
        optimizer_state_dict = optimizer.state_dict(),
        scheduler_state_dict = scheduler.state_dict(),
        scaler_state_dict    = scaler.state_dict(),
        model_cfg = asdict(m_cfg),
        train_cfg = asdict(t_cfg),
        # ✨ CHANGED: 更新模型类型元数据 (可选，但推荐)
        model_type= "latent_diffusion_v2",
    )
    torch.save(ckpt, path)
    print(f"💾  Saved → {path}")

# (load_checkpoint 无需改动)
def load_checkpoint(model, optimizer, scheduler, scaler, path: str):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt.get("best_loss", float("inf"))

# ════════════════════════════════════════════════════════════════════
#  train / eval
# (无需改动)
# ════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, scaler, device, ep):
    model.train(); tot, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {ep:03d}")
    for txt, pose, mask in pbar:
        pose, mask = pose.to(device), mask.to(device)
        with autocast(enabled=t_cfg.mixed_precision):
            loss = model(txt, pose, mask)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       t_cfg.gradient_clip_norm)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        tot += loss.item(); n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return tot / n

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); tot, n = 0.0, 0
    for txt, pose, mask in loader:
        pose, mask = pose.to(device), mask.to(device)
        loss = model(txt, pose, mask)
        tot += loss.item(); n += 1
    return tot / n

# ════════════════════════════════════════════════════════════════════
#  dataset stats (μ / σ) — 仅首次
# (无需改动)
# ════════════════════════════════════════════════════════════════════
def compute_dataset_stats(loader):
    sums = torch.zeros(m_cfg.pose_dim)
    sqs  = torch.zeros(m_cfg.pose_dim)
    count = 0
    for _, pose, mask in tqdm(loader, desc="📊 computing μ/σ"):
        valid = mask.unsqueeze(-1).expand_as(pose)
        pose_valid = pose * valid
        sums  += pose_valid.sum((0,1))
        sqs   += (pose_valid ** 2).sum((0,1))
        # 计算有效帧的总数
        count += valid.sum((0,1))
    mean = sums / count.clamp(min=1)
    var  = sqs / count.clamp(min=1) - mean ** 2
    std  = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean, std

# ════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════
def main():
    device = torch.device(m_cfg.device)
    torch.manual_seed(42); np.random.seed(42)
    print("► Starting Training for Stage 2: Latent Diffusion Model")
    print("► device:", device)

    # loaders (首次)
    tr_loader, val_loader, test_loader = create_data_loaders(data_cfg)

    # mean / std
    if m_cfg.pose_normalize and torch.all(m_cfg.std == 1):
        # 注意: compute_dataset_stats 计算的是未归一化的数据的μ/σ
        # ASLPoseDataset 在内部进行了归一化，所以我们需要从dataset实例获取
        print("✅ Retrieving μ/σ from training dataset...")
        m_cfg.mean = torch.from_numpy(tr_loader.dataset.pose_mean).float()
        m_cfg.std = torch.from_numpy(tr_loader.dataset.pose_std).float()
    else:
        print("✅ Using pre-defined μ/σ or no normalization.")


    # model & optim
    # ✨ CHANGED: 实例化新的 LatentDiffusion 模型
    model = LatentDiffusion(m_cfg).to(device)
    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(optim_params, lr=t_cfg.learning_rate, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, t_cfg, len(tr_loader))
    scaler    = GradScaler(enabled=t_cfg.mixed_precision)

    # logging
    # ... (logging 和 checkpoint 加载逻辑无需改动) ...
    os.makedirs(t_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(t_cfg.log_dir, exist_ok=True)
    writer   = SummaryWriter(t_cfg.log_dir)
    latest   = os.path.join(t_cfg.checkpoint_dir, "latest.pth")

    start_ep, best = 0, float("inf")
    if os.path.exists(latest):
        print(f"🔄  resume {latest}")
        start_ep, best = load_checkpoint(model, optimizer, scheduler, scaler, latest)
        start_ep += 1
        print(f"▶️  start epoch {start_ep}, best {best:.4f}")
    else:
        print("🆕  fresh run")

    # ... (训练主循环无需改动) ...
    no_up, patience = 0, 30
    for ep in range(start_ep, t_cfg.num_epochs):
        t0 = time.time()
        train_loss = train_epoch(model, tr_loader, optimizer, scheduler, scaler, device, ep)
        val_loss   = eval_epoch(model, val_loader, device)
        dt = time.time() - t0

        lr_now = scheduler.get_last_lr()[0]
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, ep)
        writer.add_scalar("LR", lr_now, ep)
        print(f"Epoch {ep:>4d} │ train {train_loss:.4f} │ val {val_loss:.4f} "
              f"│ lr {lr_now:.2e} │ {dt:.1f}s")

        # checkpoint
        save_checkpoint(model, optimizer, scheduler, scaler, ep,
                        train_loss, latest, best_loss=best)
        if ep % t_cfg.save_every == 0:
            periodic = os.path.join(t_cfg.checkpoint_dir, f"epoch_{ep}.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, ep,
                            train_loss, periodic, best_loss=best)
        if val_loss < best:
            best, no_up = val_loss, 0
            best_path = os.path.join(t_cfg.checkpoint_dir, "best.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, ep,
                            best, best_path, best_loss=best)
            print(f"🎯  new best {val_loss:.6f}")
        else:
            no_up += 1
            if no_up >= patience:
                print(f"⏹️  early stop (patience={patience})")
                break

    # test
    best_path = os.path.join(t_cfg.checkpoint_dir, "best.pth")
    if os.path.exists(best_path):
        print("✅  best model loaded for testing")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss = eval_epoch(model, test_loader, device)
    print(f"📊  test loss: {test_loss:.6f}")

    # demo generation
    model.eval(); print("\n🎨  sampling demo …")
    demo_texts = ["hello", "thank you", "water", "help", "good"]
    poses = model.sample(texts=demo_texts, T=60, num_steps=50) # T 可任意指定
    print(f"generated {poses.shape} | range[{poses.min():.2f},{poses.max():.2f}] "
          f"| mean {poses.mean():.2f} ± {poses.std():.2f}")

    writer.close()
    print("\n🏁  training complete")

if __name__ == "__main__":
    main()