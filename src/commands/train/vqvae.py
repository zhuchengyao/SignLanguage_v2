import argparse
import os
import time
from datetime import datetime

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from src.cli_args import add_vqvae_experiment_args, apply_vqvae_overrides
from src.config import T2M_Config
from src.model_vqvae import VQ_VAE
from src.train_utils import compute_vqvae_total_loss, create_pose_dataloaders


def codebook_perplexity(indices: torch.Tensor, codebook_size: int) -> float:
    flat_indices = indices.view(-1)
    hist = torch.bincount(flat_indices, minlength=codebook_size).float()
    probs = hist / hist.sum().clamp(min=1)
    ppl = torch.exp(-(probs * (probs + 1e-9).log()).sum())
    return float(ppl.item())


def log_memory_usage(tag: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        print(f"[{tag}] GPU Memory: Allocated={alloc:.1f}MB, Reserved={reserv:.1f}MB")


class VQVAETrainer:
    def __init__(self, cfg: T2M_Config, use_wandb: bool, resume: bool):
        self.cfg = cfg
        self.device = cfg.get_device()
        self.use_wandb = use_wandb

        self.model = VQ_VAE(cfg).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.vqvae_learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.vqvae_num_epochs, eta_min=1e-6
        )

        self.global_step = 0
        self.best_loss = float("inf")
        self.start_epoch = 0

        print(f"VQ-VAE Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        if resume:
            self.load_checkpoint()

    def create_dataloaders(self):
        train_loader, val_loader, train_dataset = create_pose_dataloaders(
            self.cfg, sampler_cache_name="train_sampler_cache.pt"
        )
        self.cfg.mean = torch.from_numpy(train_dataset.pose_mean).float()
        self.cfg.std = torch.from_numpy(train_dataset.pose_std).float()

        print(f"Train Samples: {len(train_loader.dataset)}, Val Samples: {len(val_loader.dataset)}")
        return train_loader, val_loader

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        os.makedirs(os.path.dirname(self.cfg.vqvae_checkpoint_path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "global_step": self.global_step,
            "cfg": self.cfg,
        }
        latest_path = self.cfg.vqvae_checkpoint_path.replace(".pth", "_latest.pth")
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, self.cfg.vqvae_checkpoint_path)
            print(f"Saved best model to: {self.cfg.vqvae_checkpoint_path}")

    def load_checkpoint(self):
        latest_path = self.cfg.vqvae_checkpoint_path.replace(".pth", "_latest.pth")
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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.vqvae_num_epochs}")

        for batch in pbar:
            if batch[0] is None:
                continue
            _, pose_sequences, masks = batch
            pose_sequences = pose_sequences.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(pose_sequences, masks)
            metrics = compute_vqvae_total_loss(outputs, self.cfg, epoch=epoch, use_warmup=True)
            total_loss = metrics["total_loss"]

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            try:
                ppl = codebook_perplexity(outputs["indices"], self.cfg.codebook_size)
            except Exception:
                ppl = 0.0

            ppl_coarse = 0.0
            if outputs.get("coarse_indices") is not None:
                try:
                    ppl_coarse = codebook_perplexity(
                        outputs["coarse_indices"],
                        getattr(self.cfg, "coarse_codebook_size", self.cfg.codebook_size),
                    )
                except Exception:
                    ppl_coarse = 0.0

            pbar.set_postfix(
                {
                    "Loss": f"{metrics['total_loss'].item():.4f}",
                    "Recon": f"{metrics['recon_loss'].item():.4f}",
                    "wRecon": f"{metrics['recon_weighted'].item():.4f}",
                    "Bone": f"{metrics['bone_length_loss'].item():.4f}",
                    "Vel": f"{metrics['velocity_loss'].item():.4f}",
                    "Acc": f"{metrics['accel_loss'].item():.4f}",
                    "PPL": f"{ppl:.1f}",
                    "cVQ": f"{metrics['coarse_vq_loss'].item():.4f}",
                    "cPPL": f"{ppl_coarse:.1f}",
                }
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "train/total_loss": metrics["total_loss"].item(),
                        "train/recon_loss": metrics["recon_loss"].item(),
                        "train/recon_weighted": metrics["recon_weighted"].item(),
                        "train/bone_length_loss": metrics["bone_length_loss"].item(),
                        "train/velocity_loss": metrics["velocity_loss"].item(),
                        "train/accel_loss": metrics["accel_loss"].item(),
                        "train/vq_loss": metrics["vq_loss"].item(),
                        "train/coarse_vq_loss": metrics["coarse_vq_loss"].item(),
                        "train/vq_weight": metrics["vq_weight"].item(),
                        "train/codebook_perplexity": ppl,
                        "train/coarse_codebook_perplexity": ppl_coarse,
                    },
                    step=self.global_step,
                )
            self.global_step += 1

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()

        totals = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "vq_loss": 0.0,
            "coarse_vq_loss": 0.0,
            "recon_weighted": 0.0,
            "bone_length_loss": 0.0,
            "velocity_loss": 0.0,
            "accel_loss": 0.0,
        }
        total_ppl = 0.0
        count_ppl = 0
        total_ppl_coarse = 0.0
        count_ppl_coarse = 0
        valid_batches = 0

        for batch in tqdm(val_loader, desc="Validating"):
            if batch[0] is None:
                continue

            _, pose_sequences, masks = batch
            pose_sequences = pose_sequences.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(pose_sequences, masks)
            metrics = compute_vqvae_total_loss(outputs, self.cfg, epoch=0, use_warmup=False)

            for key in totals:
                totals[key] += float(metrics[key].item())
            valid_batches += 1

            try:
                total_ppl += codebook_perplexity(outputs["indices"], self.cfg.codebook_size)
                count_ppl += 1
            except Exception:
                pass

            if outputs.get("coarse_indices") is not None:
                try:
                    total_ppl_coarse += codebook_perplexity(
                        outputs["coarse_indices"],
                        getattr(self.cfg, "coarse_codebook_size", self.cfg.codebook_size),
                    )
                    count_ppl_coarse += 1
                except Exception:
                    pass

        denom = max(valid_batches, 1)
        avg = {k: v / denom for k, v in totals.items()}
        avg_ppl = total_ppl / max(count_ppl, 1)
        avg_ppl_coarse = total_ppl_coarse / max(count_ppl_coarse, 1) if count_ppl_coarse > 0 else 0.0

        return avg, avg_ppl, avg_ppl_coarse

    def train_session(self, target_end_epoch: int):
        train_loader, val_loader = self.create_dataloaders()
        print(f"Starting VQ-VAE training session from epoch {self.start_epoch + 1}...")

        for epoch in range(self.start_epoch, target_end_epoch):
            self.train_epoch(train_loader, epoch)
            val_metrics, val_ppl, val_ppl_coarse = self.validate(val_loader)
            self.scheduler.step()

            log_memory_usage(f"End of Epoch {epoch + 1}")
            print(
                "Epoch {0} Summary: Validation Loss={1:.4f} (Recon={2:.4f}, wRecon={3:.4f}, Bone={4:.4f}, "
                "Vel={5:.4f}, Acc={6:.4f}, VQ={7:.4f}, cVQ={8:.4f}, PPL={9:.1f}, cPPL={10:.1f})".format(
                    epoch + 1,
                    val_metrics["total_loss"],
                    val_metrics["recon_loss"],
                    val_metrics["recon_weighted"],
                    val_metrics["bone_length_loss"],
                    val_metrics["velocity_loss"],
                    val_metrics["accel_loss"],
                    val_metrics["vq_loss"],
                    val_metrics["coarse_vq_loss"],
                    val_ppl,
                    val_ppl_coarse,
                )
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "val/total_loss": val_metrics["total_loss"],
                        "val/recon_loss": val_metrics["recon_loss"],
                        "val/recon_weighted": val_metrics["recon_weighted"],
                        "val/bone_length_loss": val_metrics["bone_length_loss"],
                        "val/velocity_loss": val_metrics["velocity_loss"],
                        "val/accel_loss": val_metrics["accel_loss"],
                        "val/vq_loss": val_metrics["vq_loss"],
                        "val/coarse_vq_loss": val_metrics["coarse_vq_loss"],
                        "val/codebook_perplexity": val_ppl,
                        "val/coarse_codebook_perplexity": val_ppl_coarse,
                        "epoch": epoch,
                    }
                )

            is_best = val_metrics["total_loss"] < self.best_loss
            if is_best:
                self.best_loss = val_metrics["total_loss"]
            self.save_checkpoint(epoch, val_metrics["total_loss"], is_best)

        print(f"Session complete up to epoch {target_end_epoch}!")


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE for T2M-GPT in sessions")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--session_epochs", type=int, default=3, help="Number of epochs per training session")
    add_vqvae_experiment_args(parser)
    args = parser.parse_args()

    cfg = T2M_Config()
    apply_vqvae_overrides(cfg, args)

    if args.wandb:
        wandb.init(
            project="t2m-gpt-sign-language",
            config=vars(cfg),
            name=f"vqvae_session_run_{datetime.now().strftime('%Y%m%d_%H%M')}",
            resume="allow",
            reinit=True,
        )

    start_epoch = 0
    latest_checkpoint_path = cfg.vqvae_checkpoint_path.replace(".pth", "_latest.pth")
    if os.path.exists(latest_checkpoint_path):
        print(f"Found existing checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Last completed epoch was {start_epoch - 1}. New sessions will start from here.")

    current_epoch = start_epoch
    while current_epoch < cfg.vqvae_num_epochs:
        is_resuming = (start_epoch > 0) or (current_epoch > 0)
        session_target_epoch = min(current_epoch + args.session_epochs, cfg.vqvae_num_epochs)

        print("\n" + "=" * 60)
        print(f"Starting Training Session: epoch {current_epoch + 1} -> {session_target_epoch}")
        print("=" * 60 + "\n")

        trainer = VQVAETrainer(cfg, args.wandb, resume=is_resuming)

        current_epoch = trainer.start_epoch
        session_target_epoch = min(current_epoch + args.session_epochs, cfg.vqvae_num_epochs)

        if current_epoch >= session_target_epoch:
            print(f"All epochs up to {current_epoch} already trained. Nothing to do in this session.")
            current_epoch = session_target_epoch
            continue

        try:
            trainer.train_session(target_end_epoch=session_target_epoch)
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            break

        current_epoch = session_target_epoch
        if current_epoch < cfg.vqvae_num_epochs:
            print("Simulating script restart... preparing next session.")
            time.sleep(2)

    print("All training sessions complete!")
    if args.wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
