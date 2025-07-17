"""Single‑GPU training script (automatic resume, EMA & CosineLR)."""
import json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import model_cfg, train_cfg
from dataloader import create_data_loaders  # your own impl
from motion_dit import MotionDiffusionModel  # unchanged except it picks cfg flags
from modules import EMA


def train_diffusion():
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # check VAE
    if not Path(model_cfg.vae_final_checkpoint_path).is_file():
        raise FileNotFoundError(
            f"VAE checkpoint not found → {model_cfg.vae_final_checkpoint_path}"
        )

    device = torch.device(model_cfg.get_device())
    scaler = GradScaler(enabled=train_cfg.mixed_precision)

    # ----- data loaders -----
    dl_cfg = type(
        "Tmp",
        (),
        {
            "data_root": train_cfg.data_root,
            "batch_size": train_cfg.diffusion.batch_size,
            "pose_normalize": model_cfg.pose_normalize,
            "pose_clip_range": model_cfg.pose_clip_range,
            "vae_max_seq_len": model_cfg.vae_max_seq_len,
            "num_workers": train_cfg.num_workers,
        },
    )()
    train_dl, val_dl, _ = create_data_loaders(dl_cfg)

    # load stats if available
    stats_path = Path("checkpoints/vae_stats.npz")
    if stats_path.is_file():
        stats = np.load(stats_path)
        model_cfg.mean = torch.from_numpy(stats["mean"]).float()
        model_cfg.std = torch.from_numpy(stats["std"]).float()

    # ----- model & optim -----
    model = MotionDiffusionModel(model_cfg).to(device)
    optimiser = optim.AdamW(model.noise_predictor.parameters(), lr=train_cfg.diffusion.lr)
    scheduler = CosineAnnealingLR(optimiser, T_max=train_cfg.diffusion.epochs)
    ema = EMA(model.noise_predictor.parameters(), decay=0.999)

    # resume logic
    best = math.inf
    resume_path = ckpt_dir / "diffusion_dit_best.pth"
    if resume_path.is_file():
        ckpt = torch.load(resume_path, map_location="cpu")
        model.noise_predictor.load_state_dict(ckpt["model_state_dict"])
        for s, p in zip(ema.shadow, ckpt["ema_state_dict"]):
            s.copy_(p)
        best = ckpt.get("best_val", best)
        print(f"🔄 Resumed from {resume_path} | best={best:.4f}")

    # ----- training loop -----
    for epoch in range(train_cfg.diffusion.epochs):
        model.noise_predictor.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{train_cfg.diffusion.epochs}")
        for texts, poses, masks in pbar:
            poses, masks = poses.to(device), masks.to(device)
            optimiser.zero_grad(set_to_none=True)
            with autocast(enabled=train_cfg.mixed_precision):
                loss = model(texts, poses, masks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.noise_predictor.parameters(), train_cfg.gradient_clip_norm
            )
            scaler.step(optimiser)
            scaler.update()
            ema.update(model.noise_predictor.parameters())
            pbar.set_postfix(mse=f"{loss.item():.5f}")
        scheduler.step()

        # validation
        model.noise_predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for texts, poses, masks in val_dl:
                poses, masks = poses.to(device), masks.to(device)
                val_loss += model(texts, poses, masks).item()
        val_loss /= len(val_dl)
        print(f"\n📊 epoch={epoch+1}  val={val_loss:.6f}")

        # save best
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state_dict": model.noise_predictor.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "best_val": best,
                    "cfg": model_cfg.__dict__,
                },
                resume_path,
            )
            print(f"✅  saved new best → {resume_path}")


if __name__ == "__main__":
    train_diffusion()
