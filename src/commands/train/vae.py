"""Train Motion VAE for pose sequence reconstruction."""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from src.cli_args import add_common_data_args
from src.config import T2M_Config
from src.model_vae import MotionVAE
from src.train_utils import create_pose_dataloaders


def compute_vae_total_loss(outputs, cfg, epoch):
    """Compute total VAE loss with KL warmup."""
    recon_loss = outputs["recon_loss"]
    kl_loss = outputs["kl_loss"]
    weighted_recon = outputs.get("recon_weighted", recon_loss.new_tensor(0.0))
    bone_loss = outputs.get("bone_length_loss", recon_loss.new_tensor(0.0))
    velocity_loss = outputs.get("velocity_loss", recon_loss.new_tensor(0.0))
    accel_loss = outputs.get("accel_loss", recon_loss.new_tensor(0.0))

    # KL warmup: linearly ramp from 0 to kl_weight
    kl_warmup = min(1.0, (epoch + 1) / max(1, cfg.kl_warmup_epochs))
    kl_weight = cfg.kl_weight * kl_warmup

    total_loss = (
        cfg.recon_loss_weight * recon_loss
        + kl_weight * kl_loss
        + cfg.weighted_recon_loss_weight * weighted_recon
        + cfg.bone_length_loss_weight * bone_loss
        + cfg.temporal_velocity_loss_weight * velocity_loss
        + cfg.temporal_accel_loss_weight * accel_loss
    )

    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "kl_weight": recon_loss.new_tensor(kl_weight),
        "recon_weighted": weighted_recon,
        "bone_length_loss": bone_loss,
        "velocity_loss": velocity_loss,
        "accel_loss": accel_loss,
    }


def log_memory_usage(tag: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        print(f"[{tag}] GPU Memory: Allocated={alloc:.1f}MB, Reserved={reserv:.1f}MB")


class VAETrainer:
    def __init__(self, cfg: T2M_Config, use_wandb: bool, resume: bool):
        self.cfg = cfg
        self.device = cfg.get_device()
        self.use_wandb = use_wandb

        self.model = MotionVAE(cfg).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.vae_learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.vae_num_epochs, eta_min=1e-6
        )

        self.global_step = 0
        self.best_loss = float("inf")
        self.start_epoch = 0

        print(f"Motion VAE Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Latent dim: {cfg.vae_latent_dim}")
        print(f"Device: {self.device}")
        if resume:
            self.load_checkpoint()

    def create_dataloaders(self):
        train_loader, val_loader, train_dataset = create_pose_dataloaders(
            self.cfg, sampler_cache_name="train_sampler_cache_vae.pt"
        )
        self.cfg.mean = torch.from_numpy(train_dataset.pose_mean).float()
        self.cfg.std = torch.from_numpy(train_dataset.pose_std).float()
        print(f"Train Samples: {len(train_loader.dataset)}, Val Samples: {len(val_loader.dataset)}")
        return train_loader, val_loader

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        os.makedirs(os.path.dirname(self.cfg.vae_checkpoint_path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "global_step": self.global_step,
            "cfg": self.cfg,
        }
        latest_path = self.cfg.vae_checkpoint_path.replace(".pth", "_latest.pth")
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, self.cfg.vae_checkpoint_path)
            print(f"Saved best model to: {self.cfg.vae_checkpoint_path}")

    def load_checkpoint(self):
        latest_path = self.cfg.vae_checkpoint_path.replace(".pth", "_latest.pth")
        if not os.path.exists(latest_path):
            print(f"No checkpoint found at {latest_path}. Starting from scratch.")
            return
        print(f"Resuming training from checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["val_loss"]
        print(f"Resumed from epoch {self.start_epoch}. Best loss so far: {self.best_loss:.4f}")

    def train_epoch(self, train_loader, epoch: int):
        self.model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.vae_num_epochs}")

        for batch in pbar:
            if batch[0] is None:
                continue
            _, pose_sequences, masks = batch
            pose_sequences = pose_sequences.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(pose_sequences, masks)
            metrics = compute_vae_total_loss(outputs, self.cfg, epoch=epoch)
            total_loss = metrics["total_loss"]

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            pbar.set_postfix({
                "Loss": f"{metrics['total_loss'].item():.4f}",
                "Recon": f"{metrics['recon_loss'].item():.4f}",
                "KL": f"{metrics['kl_loss'].item():.4f}",
                "klW": f"{metrics['kl_weight'].item():.4f}",
                "Bone": f"{metrics['bone_length_loss'].item():.4f}",
                "Vel": f"{metrics['velocity_loss'].item():.4f}",
            })

            if self.use_wandb:
                wandb.log({
                    "train/total_loss": metrics["total_loss"].item(),
                    "train/recon_loss": metrics["recon_loss"].item(),
                    "train/kl_loss": metrics["kl_loss"].item(),
                    "train/kl_weight": metrics["kl_weight"].item(),
                    "train/recon_weighted": metrics["recon_weighted"].item(),
                    "train/bone_length_loss": metrics["bone_length_loss"].item(),
                    "train/velocity_loss": metrics["velocity_loss"].item(),
                    "train/accel_loss": metrics["accel_loss"].item(),
                }, step=self.global_step)
            self.global_step += 1

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        totals = {
            "total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0,
            "recon_weighted": 0.0, "bone_length_loss": 0.0,
            "velocity_loss": 0.0, "accel_loss": 0.0,
        }
        valid_batches = 0

        for batch in tqdm(val_loader, desc="Validating"):
            if batch[0] is None:
                continue
            _, pose_sequences, masks = batch
            pose_sequences = pose_sequences.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(pose_sequences, masks)
            metrics = compute_vae_total_loss(outputs, self.cfg, epoch=0)

            for key in totals:
                totals[key] += float(metrics[key].item())
            valid_batches += 1

        denom = max(valid_batches, 1)
        avg = {k: v / denom for k, v in totals.items()}
        return avg

    def train_session(self, target_end_epoch: int):
        train_loader, val_loader = self.create_dataloaders()
        print(f"Starting Motion VAE training session from epoch {self.start_epoch + 1}...")

        for epoch in range(self.start_epoch, target_end_epoch):
            self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            log_memory_usage(f"End of Epoch {epoch + 1}")
            print(
                "Epoch {0} Summary: Val Loss={1:.4f} (Recon={2:.4f}, KL={3:.4f}, "
                "wRecon={4:.4f}, Bone={5:.4f}, Vel={6:.4f}, Acc={7:.4f})".format(
                    epoch + 1, val_metrics["total_loss"], val_metrics["recon_loss"],
                    val_metrics["kl_loss"], val_metrics["recon_weighted"],
                    val_metrics["bone_length_loss"], val_metrics["velocity_loss"],
                    val_metrics["accel_loss"],
                )
            )

            if self.use_wandb:
                wandb.log({
                    "val/total_loss": val_metrics["total_loss"],
                    "val/recon_loss": val_metrics["recon_loss"],
                    "val/kl_loss": val_metrics["kl_loss"],
                    "val/recon_weighted": val_metrics["recon_weighted"],
                    "val/bone_length_loss": val_metrics["bone_length_loss"],
                    "val/velocity_loss": val_metrics["velocity_loss"],
                    "val/accel_loss": val_metrics["accel_loss"],
                    "epoch": epoch,
                })

            is_best = val_metrics["total_loss"] < self.best_loss
            if is_best:
                self.best_loss = val_metrics["total_loss"]
            self.save_checkpoint(epoch, val_metrics["total_loss"], is_best)

        print(f"Session complete up to epoch {target_end_epoch}!")


def main():
    parser = argparse.ArgumentParser(description="Train Motion VAE")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--session_epochs", type=int, default=10, help="Epochs per session")
    add_common_data_args(parser)
    parser.add_argument("--vae_num_epochs", type=int, default=None)
    parser.add_argument("--vae_learning_rate", type=float, default=None)
    parser.add_argument("--vae_latent_dim", type=int, default=None)
    parser.add_argument("--kl_weight", type=float, default=None)
    parser.add_argument("--kl_warmup_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = T2M_Config()
    cfg.apply_overrides({
        "data_root": args.data_root,
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "vae_num_epochs": args.vae_num_epochs,
        "vae_learning_rate": args.vae_learning_rate,
        "vae_latent_dim": args.vae_latent_dim,
        "kl_weight": args.kl_weight,
        "kl_warmup_epochs": args.kl_warmup_epochs,
    })

    if args.wandb:
        wandb.init(
            project="t2m-sign-language-vae",
            config=vars(cfg),
            name=f"vae_run_{datetime.now().strftime('%Y%m%d_%H%M')}",
            resume="allow", reinit=True,
        )

    start_epoch = 0
    latest_checkpoint_path = cfg.vae_checkpoint_path.replace(".pth", "_latest.pth")
    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
        start_epoch = checkpoint.get("epoch", -1) + 1

    current_epoch = start_epoch
    while current_epoch < cfg.vae_num_epochs:
        is_resuming = current_epoch > 0
        session_target = min(current_epoch + args.session_epochs, cfg.vae_num_epochs)

        print("\n" + "=" * 60)
        print(f"Starting VAE Training Session: epoch {current_epoch + 1} -> {session_target}")
        print("=" * 60 + "\n")

        trainer = VAETrainer(cfg, args.wandb, resume=is_resuming)
        current_epoch = trainer.start_epoch
        session_target = min(current_epoch + args.session_epochs, cfg.vae_num_epochs)

        if current_epoch >= session_target:
            current_epoch = session_target
            continue

        try:
            trainer.train_session(target_end_epoch=session_target)
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break

        current_epoch = session_target
        if current_epoch < cfg.vae_num_epochs:
            time.sleep(2)

    print("All training sessions complete!")
    if args.wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
