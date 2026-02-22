"""Train Dual-branch Motion VAE (body + hand decoupled)."""

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
from src.model_dual_vae import DualMotionVAE
from src.train_utils import create_pose_dataloaders


def compute_dual_loss(outputs, cfg, epoch):
    """Compute total loss for Dual VAE with KL warmup."""
    kl_warmup = min(1.0, (epoch + 1) / max(1, cfg.dual_kl_warmup_epochs))
    kl_w = cfg.dual_kl_weight * kl_warmup

    total = (
        cfg.dual_body_recon_weight * outputs['body_recon_loss']
        + cfg.dual_hand_recon_weight * outputs['hand_recon_loss']
        + kl_w * (outputs['kl_body'] + outputs['kl_hand'])
        + cfg.dual_sync_weight * outputs['sync_loss']
        + cfg.dual_body_vel_weight * outputs['body_vel']
        + cfg.dual_body_acc_weight * outputs['body_acc']
        + cfg.dual_hand_vel_weight * outputs['hand_vel']
        + cfg.dual_hand_acc_weight * outputs['hand_acc']
    )
    return total, kl_w


class DualVAETrainer:
    def __init__(self, cfg: T2M_Config, use_wandb: bool = False, resume: bool = False):
        self.cfg = cfg
        self.use_wandb = use_wandb
        self.device = cfg.get_device()

        self.model = DualMotionVAE(cfg).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"DualMotionVAE parameters: {param_count:,}")
        print(f"  Body latent: {cfg.dual_body_latent_dim}, Hand latent: {cfg.dual_hand_latent_dim}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.dual_vae_learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.dual_vae_num_epochs, eta_min=1e-6)
        self.best_loss = float('inf')
        self.start_epoch = 0

        latest_ckpt = cfg.dual_vae_checkpoint_path.replace(".pth", "_latest.pth")
        if resume and os.path.exists(latest_ckpt):
            print(f"Resuming from: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.start_epoch = ckpt.get('epoch', -1) + 1
            self.best_loss = ckpt.get('best_loss', float('inf'))
            print(f"Resumed from epoch {self.start_epoch}. Best loss: {self.best_loss:.4f}")

        train_loader, val_loader, train_ds = create_pose_dataloaders(cfg, "train_sampler_cache_dual_vae.pt")
        self.train_loader = train_loader
        self.val_loader = val_loader

        cfg.mean = train_ds.mean_tensor
        cfg.std = train_ds.std_tensor
        print(f"Train: {len(train_ds)}, Val: {len(val_loader.dataset)}")

    def save_checkpoint(self, epoch, val_loss, is_best):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': min(self.best_loss, val_loss),
            'cfg': self.cfg,
        }
        os.makedirs(os.path.dirname(self.cfg.dual_vae_checkpoint_path), exist_ok=True)
        latest = self.cfg.dual_vae_checkpoint_path.replace(".pth", "_latest.pth")
        torch.save(ckpt, latest)
        if is_best:
            torch.save(ckpt, self.cfg.dual_vae_checkpoint_path)
            print(f"Saved best model to: {self.cfg.dual_vae_checkpoint_path}")

    def run_epoch(self, loader, epoch, train=True):
        self.model.train() if train else self.model.eval()
        total_metrics = {}
        n = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'} {epoch+1}/{self.cfg.dual_vae_num_epochs}")

        with ctx:
            for batch in pbar:
                texts, poses, masks = batch
                if poses is None:
                    continue
                poses, masks = poses.to(self.device), masks.to(self.device)

                outputs = self.model(poses, masks)
                loss, kl_w = compute_dual_loss(outputs, self.cfg, epoch)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                B = poses.size(0)
                for k in ['body_recon_loss', 'hand_recon_loss', 'kl_body', 'kl_hand',
                           'sync_loss', 'body_vel', 'body_acc', 'hand_vel', 'hand_acc']:
                    total_metrics[k] = total_metrics.get(k, 0.0) + outputs[k].item() * B
                total_metrics['total'] = total_metrics.get('total', 0.0) + loss.item() * B
                n += B

                if train:
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.3f}',
                        'Body': f'{outputs["body_recon_loss"].item():.3f}',
                        'Hand': f'{outputs["hand_recon_loss"].item():.3f}',
                        'Sync': f'{outputs["sync_loss"].item():.3f}',
                        'klW': f'{kl_w:.4f}',
                    })

        return {k: v / max(n, 1) for k, v in total_metrics.items()}

    def train_session(self, target_end_epoch: int):
        for epoch in range(self.start_epoch, target_end_epoch):
            train_m = self.run_epoch(self.train_loader, epoch, train=True)
            val_m = self.run_epoch(self.val_loader, epoch, train=False)
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}: Val Total={val_m['total']:.4f} "
                f"(Body={val_m['body_recon_loss']:.4f}, Hand={val_m['hand_recon_loss']:.4f}, "
                f"Sync={val_m['sync_loss']:.4f}, KL_b={val_m['kl_body']:.4f}, KL_h={val_m['kl_hand']:.4f})"
            )

            if self.use_wandb:
                wandb.log({f"train/{k}": v for k, v in train_m.items()} |
                          {f"val/{k}": v for k, v in val_m.items()} |
                          {"epoch": epoch})

            is_best = val_m['total'] < self.best_loss
            if is_best:
                self.best_loss = val_m['total']
            self.save_checkpoint(epoch, val_m['total'], is_best)

        print(f"Session complete up to epoch {target_end_epoch}!")


def main():
    parser = argparse.ArgumentParser(description="Train Dual-branch Motion VAE")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--session_epochs", type=int, default=10)
    add_common_data_args(parser)
    parser.add_argument("--dual_vae_num_epochs", type=int, default=None)
    parser.add_argument("--dual_vae_learning_rate", type=float, default=None)
    args = parser.parse_args()

    cfg = T2M_Config()
    cfg.apply_overrides({
        "data_root": args.data_root,
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "dual_vae_num_epochs": args.dual_vae_num_epochs,
        "dual_vae_learning_rate": args.dual_vae_learning_rate,
    })

    if args.wandb:
        wandb.init(project="t2m-dual-vae", config=vars(cfg),
                   name=f"dual_vae_{datetime.now().strftime('%Y%m%d_%H%M')}", resume="allow", reinit=True)

    current_epoch = 0
    latest_ckpt = cfg.dual_vae_checkpoint_path.replace(".pth", "_latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        current_epoch = ckpt.get("epoch", -1) + 1

    while current_epoch < cfg.dual_vae_num_epochs:
        session_target = min(current_epoch + args.session_epochs, cfg.dual_vae_num_epochs)
        print(f"\n{'='*60}")
        print(f"Dual VAE Session: epoch {current_epoch+1} -> {session_target}")
        print(f"{'='*60}\n")

        trainer = DualVAETrainer(cfg, args.wandb, resume=(current_epoch > 0))
        current_epoch = trainer.start_epoch
        session_target = min(current_epoch + args.session_epochs, cfg.dual_vae_num_epochs)

        if current_epoch >= session_target:
            current_epoch = session_target
            continue

        try:
            trainer.train_session(target_end_epoch=session_target)
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        current_epoch = session_target
        if current_epoch < cfg.dual_vae_num_epochs:
            time.sleep(2)

    print("All training sessions complete!")
    if args.wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
