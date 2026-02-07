import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from transformers import BertTokenizer

from src.cli_args import add_gpt_experiment_args, apply_gpt_overrides
from src.config import T2M_Config
from src.model_gpt import T2M_GPT
from src.model_vqvae import VQ_VAE
from src.train_utils import create_pose_dataloaders


class GPTTrainer:
    def __init__(self, cfg: T2M_Config, use_wandb: bool):
        self.cfg = cfg
        self.device = cfg.get_device()
        self.use_wandb = use_wandb

        print(f"Loading pre-trained VQ-VAE from: {cfg.vqvae_checkpoint_path}")
        vqvae_checkpoint = torch.load(cfg.vqvae_checkpoint_path, map_location="cpu", weights_only=False)
        self.vq_cfg = vqvae_checkpoint["cfg"]
        self.vq_vae = VQ_VAE(self.vq_cfg).to(self.device)
        self.vq_vae.load_state_dict(vqvae_checkpoint["model_state_dict"], strict=False)
        self.vq_vae.eval()
        for param in self.vq_vae.parameters():
            param.requires_grad = False

        # Keep GPT vocab/sequence settings aligned with the VQ-VAE used for tokenization.
        self.cfg.codebook_size = self.vq_cfg.codebook_size
        self.cfg.downsample_rate = self.vq_cfg.downsample_rate
        self.cfg.model_max_seq_len = self.vq_cfg.model_max_seq_len
        self.cfg.refresh_dependent_values()

        if getattr(self.vq_cfg, "use_hierarchical_codebook", False):
            print(
                "Warning: loaded VQ-VAE uses hierarchical codebook, "
                "but GPT currently models fine tokens only."
            )

        self.model = T2M_GPT(cfg).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.gpt_learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.gpt_num_epochs)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.global_step = 0
        self.best_loss = float("inf")

        print("VQ-VAE loaded and frozen.")
        print(f"T2M-GPT Model Parameters (Trainable): {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def create_dataloaders(self):
        train_loader, val_loader, train_dataset = create_pose_dataloaders(
            self.cfg, sampler_cache_name="gpt_train_sampler_cache.pt"
        )
        self.cfg.mean = torch.from_numpy(train_dataset.pose_mean).float()
        self.cfg.std = torch.from_numpy(train_dataset.pose_std).float()
        return train_loader, val_loader

    @torch.no_grad()
    def encode_motion_to_tokens(self, poses, masks):
        _, indices, _ = self.vq_vae.encode(poses, masks)
        return indices

    def _build_teacher_forcing_batch(self, gt_motion_tokens: torch.Tensor, pose_masks: torch.Tensor):
        bsz, t_down = gt_motion_tokens.shape
        num_valid_poses = pose_masks.sum(dim=1)
        num_valid_tokens = torch.ceil(num_valid_poses / self.cfg.downsample_rate).long()
        motion_token_mask = (
            torch.arange(t_down, device=self.device).expand(bsz, t_down) < num_valid_tokens.unsqueeze(1)
        )

        sos_token = self.cfg.codebook_size
        eos_token = self.cfg.codebook_size + 1 if getattr(self.cfg, "use_eos_token", True) else None
        sos_tensor = torch.full((bsz, 1), sos_token, device=self.device, dtype=torch.long)

        input_tokens = torch.cat([sos_tensor, gt_motion_tokens[:, :-1]], dim=1)
        target_tokens = gt_motion_tokens.clone()
        input_mask = torch.cat([torch.ones_like(sos_tensor, dtype=torch.bool), motion_token_mask[:, :-1]], dim=1)

        if eos_token is not None:
            first_invalid = (~motion_token_mask).float().argmax(dim=1)
            for b in range(bsz):
                pos = int(first_invalid[b].item())
                if motion_token_mask[b, -1]:
                    target_tokens[b, -1] = eos_token
                else:
                    target_tokens[b, pos] = eos_token

        target_tokens[~motion_token_mask] = self.criterion.ignore_index
        return input_tokens, target_tokens, input_mask

    def train_epoch(self, train_loader, epoch: int):
        self.model.train()
        self.model.text_encoder.eval()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.gpt_num_epochs}")

        for texts, pose_sequences, pose_masks in pbar:
            if texts is None:
                continue

            pose_sequences = pose_sequences.to(self.device)
            pose_masks = pose_masks.to(self.device)
            gt_motion_tokens = self.encode_motion_to_tokens(pose_sequences, pose_masks)

            input_tokens, target_tokens, input_mask = self._build_teacher_forcing_batch(gt_motion_tokens, pose_masks)
            tokenized_text = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)

            predicted_logits = self.model(tokenized_text, input_tokens, input_mask)
            loss = self.criterion(predicted_logits.reshape(-1, predicted_logits.size(-1)), target_tokens.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            if self.use_wandb:
                wandb.log({"train/gpt_loss": loss.item()}, step=self.global_step)
            self.global_step += 1

    @torch.no_grad()
    def validate(self, val_loader, epoch: int):
        self.model.eval()
        total_loss = 0.0
        valid_batches = 0

        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}")
        for texts, pose_sequences, pose_masks in pbar:
            if texts is None:
                continue

            pose_sequences = pose_sequences.to(self.device)
            pose_masks = pose_masks.to(self.device)
            gt_motion_tokens = self.encode_motion_to_tokens(pose_sequences, pose_masks)

            input_tokens, target_tokens, input_mask = self._build_teacher_forcing_batch(gt_motion_tokens, pose_masks)
            tokenized_text = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)

            predicted_logits = self.model(tokenized_text, input_tokens, input_mask)
            loss = self.criterion(predicted_logits.reshape(-1, predicted_logits.size(-1)), target_tokens.reshape(-1))

            total_loss += float(loss.item())
            valid_batches += 1

        avg_loss = total_loss / max(valid_batches, 1)
        if self.use_wandb:
            wandb.log({"val/gpt_loss": avg_loss, "epoch": epoch})
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        os.makedirs(os.path.dirname(self.cfg.gpt_checkpoint_path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "cfg": self.cfg,
        }
        latest_path = self.cfg.gpt_checkpoint_path.replace(".pth", "_latest.pth")
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, self.cfg.gpt_checkpoint_path)
            print(f"Saved best GPT model to: {self.cfg.gpt_checkpoint_path}")

    def train(self):
        train_loader, val_loader = self.create_dataloaders()
        for epoch in range(self.cfg.gpt_num_epochs):
            self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            print(f"Epoch {epoch + 1} Summary: Validation Loss = {val_loss:.4f}")

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best=is_best)
            self.scheduler.step()

        print(f"GPT training complete. Best validation loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train T2M-GPT (Phase 2)")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb even if --wandb is set")
    add_gpt_experiment_args(parser)
    args = parser.parse_args()

    cfg = T2M_Config()
    apply_gpt_overrides(cfg, args)

    if args.wandb and not args.no_wandb:
        wandb.init(
            project="t2m-gpt-sign-language",
            name=f"gpt_{datetime.now().strftime('%Y%m%d_%H%M')}",
            config=vars(cfg),
        )

    trainer = GPTTrainer(cfg, use_wandb=(args.wandb and not args.no_wandb))
    trainer.train()


if __name__ == "__main__":
    main()
