"""
Small-scale GPT training on a subset of ASL_sentence dataset.

Usage:
    python train_gpt_sentence.py --num_train 200 --num_val 50 --epochs 10
    python train_gpt_sentence.py --num_train 500 --num_val 100 --epochs 20 --wandb
"""
import os
import sys
import random
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import BertTokenizer

from src.config import T2M_Config
from src.model_vqvae import VQ_VAE
from src.model_gpt import T2M_GPT
from src.dataloader import ASLPoseDataset, collate_pose_batch


def sample_indices(total: int, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), min(n, total)))


class SentenceGPTTrainer:
    def __init__(self, cfg: T2M_Config, args):
        self.cfg = cfg
        self.args = args
        self.device = cfg.get_device()
        self.use_wandb = args.wandb

        # ---- VQ-VAE (frozen) ----
        print(f"Loading pre-trained VQ-VAE from: {cfg.vqvae_checkpoint_path}")
        vq_ck = torch.load(cfg.vqvae_checkpoint_path, map_location="cpu", weights_only=False)
        self.vq_vae = VQ_VAE(vq_ck["cfg"]).to(self.device)
        self.vq_vae.load_state_dict(vq_ck["model_state_dict"], strict=False)
        self.vq_vae.eval()
        for p in self.vq_vae.parameters():
            p.requires_grad = False
        print("VQ-VAE loaded and frozen.")

        # ---- GPT ----
        self.model = T2M_GPT(cfg).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.global_step = 0
        self.best_val_loss = float("inf")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"GPT trainable parameters: {trainable:,}")

    def build_dataloaders(self):
        train_root = os.path.join(self.cfg.data_root, "ASL_sentence/train")
        test_root = os.path.join(self.cfg.data_root, "ASL_sentence/test")

        full_train = ASLPoseDataset(
            data_paths=[train_root], split="train",
            max_seq_len=self.cfg.model_max_seq_len,
        )
        full_test = ASLPoseDataset(
            data_paths=[test_root], split="test",
            extern_mean=full_train.pose_mean,
            extern_std=full_train.pose_std,
            max_seq_len=self.cfg.model_max_seq_len,
        )

        train_ids = sample_indices(len(full_train), self.args.num_train, seed=self.args.seed)
        val_ids = sample_indices(len(full_test), self.args.num_val, seed=self.args.seed)

        train_subset = Subset(full_train, train_ids)
        val_subset = Subset(full_test, val_ids)

        print(f"\n=== Dataset summary ===")
        print(f"  Train subset: {len(train_subset)} / {len(full_train)}")
        print(f"  Val   subset: {len(val_subset)} / {len(full_test)}")

        # preview a few texts
        print(f"\n--- Sample texts (train) ---")
        for i in train_ids[:5]:
            item = full_train[i]
            if item:
                txt = item[0][:80] + ("..." if len(item[0]) > 80 else "")
                print(f"  [{i:>5d}] {txt}")

        loader_kwargs = dict(
            collate_fn=collate_pose_batch,
            num_workers=0,
            pin_memory=False,
        )
        train_loader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_subset, batch_size=self.args.batch_size, shuffle=False, **loader_kwargs)
        return train_loader, val_loader

    @torch.no_grad()
    def encode_to_tokens(self, poses, masks):
        _, indices, _ = self.vq_vae.encode(poses, masks)
        return indices

    def train_epoch(self, loader, epoch):
        self.model.train()
        self.model.text_encoder.eval()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{self.args.epochs}")

        for texts, poses, masks in pbar:
            if texts is None:
                continue
            poses, masks = poses.to(self.device), masks.to(self.device)
            gt_tokens = self.encode_to_tokens(poses, masks)

            B, T_down = gt_tokens.shape
            n_valid = torch.ceil(masks.sum(1) / self.cfg.downsample_rate).long()
            token_mask = torch.arange(T_down, device=self.device).expand(B, T_down) < n_valid.unsqueeze(1)

            sos = self.cfg.codebook_size
            eos = self.cfg.codebook_size + 1 if getattr(self.cfg, "use_eos_token", True) else None
            sos_t = torch.full((B, 1), sos, device=self.device, dtype=torch.long)

            inp = torch.cat([sos_t, gt_tokens[:, :-1]], dim=1)
            tgt = gt_tokens.clone()
            inp_mask = torch.cat([torch.ones_like(sos_t, dtype=torch.bool), token_mask[:, :-1]], dim=1)

            if eos is not None:
                first_inv = (~token_mask).float().argmax(dim=1)
                for b in range(B):
                    if token_mask[b, -1]:
                        tgt[b, -1] = eos
                    else:
                        tgt[b, int(first_inv[b].item())] = eos

            tok_text = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            logits = self.model(tok_text, inp, inp_mask)

            tgt[~token_mask] = self.criterion.ignore_index
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.global_step += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        total_loss, n_batches = 0.0, 0

        for texts, poses, masks in tqdm(loader, desc=f"Val Epoch {epoch+1}"):
            if texts is None:
                continue
            poses, masks = poses.to(self.device), masks.to(self.device)
            gt_tokens = self.encode_to_tokens(poses, masks)

            B, T_down = gt_tokens.shape
            n_valid = torch.ceil(masks.sum(1) / self.cfg.downsample_rate).long()
            token_mask = torch.arange(T_down, device=self.device).expand(B, T_down) < n_valid.unsqueeze(1)

            sos = self.cfg.codebook_size
            eos = self.cfg.codebook_size + 1 if getattr(self.cfg, "use_eos_token", True) else None
            sos_t = torch.full((B, 1), sos, device=self.device, dtype=torch.long)

            inp = torch.cat([sos_t, gt_tokens[:, :-1]], dim=1)
            tgt = gt_tokens.clone()
            inp_mask = torch.cat([torch.ones_like(sos_t, dtype=torch.bool), token_mask[:, :-1]], dim=1)

            if eos is not None:
                first_inv = (~token_mask).float().argmax(dim=1)
                for b in range(B):
                    if token_mask[b, -1]:
                        tgt[b, -1] = eos
                    else:
                        tgt[b, int(first_inv[b].item())] = eos

            tok_text = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            logits = self.model(tok_text, inp, inp_mask)

            tgt[~token_mask] = self.criterion.ignore_index
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        ckpt_dir = os.path.join("checkpoints", "sentence")
        os.makedirs(ckpt_dir, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "cfg": self.cfg,
            "args": vars(self.args),
        }
        torch.save(state, os.path.join(ckpt_dir, "gpt_sentence_latest.pth"))
        if is_best:
            torch.save(state, os.path.join(ckpt_dir, "gpt_sentence_best.pth"))
            print(f"  -> Saved BEST model (val_loss={val_loss:.4f})")

    def run(self):
        train_loader, val_loader = self.build_dataloaders()

        if self.use_wandb:
            import wandb
            wandb.init(
                project="t2m-gpt-sentence",
                name=f"sent_{self.args.num_train}tr_{datetime.now().strftime('%m%d_%H%M')}",
                config=vars(self.args),
            )

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best)
            self.scheduler.step()

            print(f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                  f"  {'*BEST*' if is_best else ''}\n")

            if self.use_wandb:
                import wandb
                wandb.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})

        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: checkpoints/sentence/")


def main():
    parser = argparse.ArgumentParser(description="Small-scale GPT training on ASL_sentence subset")
    parser.add_argument("--num_train", type=int, default=200, help="Number of training samples to use")
    parser.add_argument("--num_val", type=int, default=50, help="Number of validation samples to use")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = T2M_Config()
    trainer = SentenceGPTTrainer(cfg, args)
    trainer.run()


if __name__ == "__main__":
    main()
