# train_vae.py (Corrected Config Passing)
import os
import torch
import torch.optim as optim
import argparse
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from types import SimpleNamespace

# 从config导入已经实例化的配置对象
from config import model_cfg, train_cfg
from dataloader import create_data_loaders
# 假设你的VAE模型在一个名为 motion_vae.py 的文件中
from motion_vae import GraphTransformerVAE

KL_SAVE_THRESHOLD = 400.0

def vae_loss_function(recon_x, x, mu, logvar, mask, kl_beta: float, velocity_weight: float):
    recon_loss = (recon_x - x).pow(2).mean(dim=-1)
    recon_loss = (recon_loss * mask).sum() / mask.sum().clamp(min=1)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    velocity_loss = torch.tensor(0.0, device=x.device)
    if velocity_weight > 0 and x.size(1) > 1:
        pred_velocity = recon_x[:, 1:] - recon_x[:, :-1]
        true_velocity = x[:, 1:] - x[:, :-1]
        velocity_mask = mask[:, 1:] & mask[:, :-1]
        mse_v = (pred_velocity - true_velocity).pow(2).mean(-1)
        velocity_loss = (mse_v * velocity_mask).sum() / velocity_mask.sum().clamp(min=1)
    total_loss = recon_loss + kl_beta * kld_loss + velocity_weight * velocity_loss
    return total_loss, recon_loss, kld_loss, velocity_loss

@torch.no_grad()
def eval_epoch(model, loader, device, stage_cfg):
    model.eval()
    total_loss, total_recon, total_kl, total_velo = 0, 0, 0, 0
    # ✨ FIX: 确保dev_loader不为空
    if len(loader) == 0:
        print("Warning: dev_loader is empty. Skipping validation.")
        return {"total": 0, "recon": 0, "kl": 0, "velo": 0}
        
    for batch in tqdm(loader, desc="  [Validating]"):
        if not batch or not batch[0]: continue
        _, poses, masks = batch
        poses, masks = poses.to(device), masks.to(device)
        
        recon_poses, mu, logvar = model(poses, masks)
        loss, recon, kl, vel = vae_loss_function(
            recon_poses, poses, mu, logvar, masks,
            kl_beta=stage_cfg.kl_beta,
            velocity_weight=stage_cfg.velocity_weight
        )
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_velo += vel.item() if isinstance(vel, torch.Tensor) else vel

    num_batches = len(loader)
    return {
        "total": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "kl": total_kl / num_batches,
        "velo": total_velo / num_batches,
    }

def train_vae(stage: int):
    # --- 1. 根据阶段选择配置 ---
    if stage == 1:
        print("🚀 Starting VAE Training: STAGE 1 (Reconstruction-only)...")
        stage_cfg = train_cfg.vae_stage1
        load_ckpt_path = None
        save_ckpt_path = model_cfg.vae_recon_checkpoint_path
        best_loss_metric = 'recon'
    elif stage == 2:
        print("🚀 Starting VAE Training: STAGE 2 (Fine-tuning Regularization)...")
        stage_cfg = train_cfg.vae_stage2
        load_ckpt_path = model_cfg.vae_recon_checkpoint_path
        save_ckpt_path = model_cfg.vae_final_checkpoint_path
        best_loss_metric = 'total'
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")

    # --- 2. 设置 ---
    device = torch.device(model_cfg.get_device())
    scaler = GradScaler(enabled=train_cfg.mixed_precision)

    # ✨ FIXED: 手动组装一个包含所有必需参数的完整配置对象，传递给 data_loader
    data_loader_cfg = SimpleNamespace(
        # From model_cfg
        pose_normalize=model_cfg.pose_normalize,
        pose_clip_range=model_cfg.pose_clip_range,
        vae_max_seq_len=model_cfg.vae_max_seq_len,
        # From train_cfg
        data_root=train_cfg.data_root,
        num_workers=train_cfg.num_workers,
        bucket_boundaries=getattr(train_cfg, 'bucket_boundaries', [50, 100, 150, 200]),
        # From stage_cfg
        batch_size=stage_cfg.batch_size
    )
    train_loader, dev_loader, _ = create_data_loaders(data_loader_cfg)

    # --- 3. 模型, 优化器, 调度器 ---
    model = GraphTransformerVAE(model_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=stage_cfg.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    if load_ckpt_path and os.path.exists(load_ckpt_path):
        print(f"🔄 Loading weights from: {load_ckpt_path}")
        model.load_state_dict(torch.load(load_ckpt_path, map_location=device))
        print("   -> Weights loaded successfully.")
    elif stage == 2:
        raise FileNotFoundError(f"Checkpoint for Stage 2 not found at {load_ckpt_path}. Please run Stage 1 first.")

    best_val_loss = float('inf')
    kl_threshold_crossed = False

    # --- 4. 训练循环 ---
    for epoch in range(stage_cfg.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{stage_cfg.epochs} [Stage {stage}]")
        
        for batch in progress_bar:
            if not batch or not batch[0]: continue
            _, poses, masks = batch; poses, masks = poses.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=train_cfg.mixed_precision):
                recon_poses, mu, logvar = model(poses, masks)
                loss, recon, kl, vel = vae_loss_function(recon_poses, poses, mu, logvar, masks, kl_beta=stage_cfg.kl_beta, velocity_weight=stage_cfg.velocity_weight)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            progress_bar.set_postfix_str(f"lr={optimizer.param_groups[0]['lr']:.1e}, loss={loss.item():.4f}, recon={recon.item():.4f}, kl={kl.item():.2f}, vel={vel.item():.4f}")

        val_losses = eval_epoch(model, dev_loader, device, stage_cfg)
        print(f"Epoch {epoch+1} Summary | Avg Val Loss: {val_losses['total']:.4f} [Recon: {val_losses['recon']:.4f}, KL: {val_losses['kl']:.2f}, Velo: {val_losses['velo']:.4f}]")
        
        current_val_loss_for_scheduler = val_losses[best_loss_metric]
        scheduler.step(current_val_loss_for_scheduler)

        if stage == 1 and val_losses['recon'] < best_val_loss:
            best_val_loss = val_losses['recon']
            torch.save(model.state_dict(), save_ckpt_path)
            print(f"✅ Saved new best Stage 1 model to {save_ckpt_path} (Recon Loss: {best_val_loss:.4f})")
        
        elif stage == 2:
            if not kl_threshold_crossed and val_losses['kl'] < KL_SAVE_THRESHOLD:
                kl_threshold_crossed = True
                best_val_loss = float('inf') 
                print(f"✨ KL Divergence threshold ({KL_SAVE_THRESHOLD}) reached! Best loss tracker has been reset. Now searching for the best overall model.")

            if kl_threshold_crossed and val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save(model.state_dict(), save_ckpt_path)
                print(f"🎯 Saved new best Stage 2 model to {save_ckpt_path} (Total Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VAE training in two stages.")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2], help="The training stage to run (1 or 2).")
    args = parser.parse_args()
    if train_cfg.train_vae:
        train_vae(stage=args.stage)