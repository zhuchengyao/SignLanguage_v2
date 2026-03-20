"""Overfit training on 15 fixed ASL_sentence cases (Stage 1 → 2 → 3 in one script).

Target: drive train loss to near-zero on these 15 specific sentences so that
inference reproduces each one faithfully.

Key overfit-friendly changes vs. default config:
  - batch_size = 5 (3 batches per epoch for 15 samples)
  - vq_ema_decay = 0.90  (faster codebook adaptation)
  - vq_warmup_epochs = 3
  - length_pred_loss_weight = 0.5  (was 0.01 — ensures length predictor converges)
  - kals weights halved  (relax geometric constraints so loss can go lower)
  - stage1 500 epochs, stage2 200 epochs, stage3 100 epochs
  - train == val (same 15 samples; we WANT val_loss == train_loss == 0)
  - num_workers = 0
"""

import gc
import math
import os
import json
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from src.config_hlc import HLC_NAR_Config
from src.model_unified import HLC_NAR_Model
from src.rtp import compute_gt_rhythm


# ---------------------------------------------------------------------------
# Hardcoded 15 cases  (ASL_sentence/train)
# ---------------------------------------------------------------------------
CASE_IDS = [
    "dev_--8pSDeC-fg_1-5-rgb_front",
    "dev_--8pSDeC-fg_11-5-rgb_front",
    "dev_--8pSDeC-fg_5-5-rgb_front",
    "dev_--8pSDeC-fg_6-5-rgb_front",
    "dev_--8pSDeC-fg_7-5-rgb_front",
    "dev_--8pSDeC-fg_8-5-rgb_front",
    "dev_--dANj_01AU_10-5-rgb_front",
    "dev_--dANj_01AU_11-5-rgb_front",
    "dev_--dANj_01AU_12-5-rgb_front",
    "dev_--dANj_01AU_13-5-rgb_front",
    "dev_--dANj_01AU_16-5-rgb_front",
    "dev_--dANj_01AU_19-5-rgb_front",
    "dev_--dANj_01AU_2-5-rgb_front",
    "dev_--dANj_01AU_20-5-rgb_front",
    "dev_--dANj_01AU_3-5-rgb_front",
]
DATA_ROOT = "datasets/ASL_sentence/train"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OverfitDataset(Dataset):
    """Fixed dataset of 15 samples.  Computes and caches mean/std internally."""

    def __init__(self, case_ids, data_root, max_seq_len=200,
                 extern_mean=None, extern_std=None):
        self.samples = []
        raw_seqs = []

        for sid in case_ids:
            base = os.path.join(data_root, sid)
            text_f = os.path.join(base, "text.txt")
            pose_f = os.path.join(base, "pose.json")
            if not (os.path.exists(text_f) and os.path.exists(pose_f)):
                print(f"  [WARN] missing: {base}")
                continue
            with open(text_f, encoding="utf-8") as f:
                text = f.read().strip()
            with open(pose_f, encoding="utf-8") as f:
                js = json.load(f)
            frames = js.get("poses", [])
            seq = [
                fr.get("pose_keypoints_2d", [])
                + fr.get("hand_right_keypoints_2d", [])
                + fr.get("hand_left_keypoints_2d", [])
                for fr in frames
            ]
            seq = [p for p in seq if len(p) == 150]
            if not seq:
                print(f"  [WARN] empty pose: {sid}")
                continue
            arr = np.array(seq, dtype=np.float32)
            if max_seq_len and len(arr) > max_seq_len:
                arr = arr[:max_seq_len]
            raw_seqs.append(arr)
            self.samples.append((text, arr))

        # Compute normalisation stats from these 15 samples
        if extern_mean is not None and extern_std is not None:
            self.pose_mean = extern_mean
            self.pose_std = extern_std
        else:
            all_frames = np.concatenate(raw_seqs, axis=0)
            self.pose_mean = all_frames.mean(axis=0)
            self.pose_std = all_frames.std(axis=0)
            self.pose_std[self.pose_std < 1e-8] = 1e-8

        # Normalise and convert to tensors (wider clamp to preserve extremes)
        self._items = []
        for text, arr in self.samples:
            t = torch.from_numpy((arr - self.pose_mean) / self.pose_std).float()
            t = t.clamp(-5.0, 5.0)
            self._items.append((text, t))

        print(f"OverfitDataset: {len(self._items)} samples loaded.")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None
    texts, seqs = zip(*batch)
    lens = [s.size(0) for s in seqs]
    T_max = max(lens)
    padded, masks = [], []
    for s, L in zip(seqs, lens):
        pad = T_max - L
        padded.append(torch.cat([s, s.new_zeros(pad, s.size(-1))], dim=0))
        masks.append(torch.cat([torch.ones(L, dtype=torch.bool),
                                 torch.zeros(pad, dtype=torch.bool)]))
    return list(texts), torch.stack(padded), torch.stack(masks)


# ---------------------------------------------------------------------------
# Overfit config overrides
# ---------------------------------------------------------------------------

def make_overfit_cfg():
    cfg = HLC_NAR_Config()
    # Faster codebook convergence
    cfg.vq_ema_decay = 0.85
    cfg.vq_warmup_epochs = 2
    # Smaller codebooks — 15 samples don't need huge codebooks
    cfg.global_codebook_size = 256
    cfg.local_codebook_size = 512
    # Length predictor must converge
    cfg.length_pred_loss_weight = 1.0
    # Disable geometric constraints entirely — let recon go as low as possible
    cfg.kals_bone_weight = 0.0
    cfg.kals_angle_weight = 0.0
    cfg.kals_symmetry_weight = 0.0
    # Boost reconstruction weight
    cfg.recon_loss_weight = 5.0
    cfg.body_recon_weight = 1.0
    cfg.hand_recon_weight = 1.5
    # VQ loss weight — keep low so recon dominates
    cfg.vq_loss_weight = 0.1
    # Token pred
    cfg.token_pred_loss_weight = 2.0
    # RTP
    cfg.rtp_loss_weight = 0.05
    # Epochs — aggressive overfit
    cfg.stage1_epochs = 800
    cfg.stage2_epochs = 400
    cfg.stage3_epochs = 300
    # Higher LR for faster convergence on tiny dataset
    cfg.stage1_lr = 5e-4
    cfg.stage2_lr = 3e-4
    cfg.stage3_lr = 1e-4
    # Batch / workers
    cfg.stage1_batch_size = 5
    cfg.stage2_batch_size = 5
    cfg.stage3_batch_size = 5
    cfg.num_workers = 0
    cfg.pin_memory = False
    return cfg


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def train_stage1(model, loader, cfg, device, tokenizer):
    print("\n" + "="*60)
    print("STAGE 1: HLC reconstruction warm-up")
    print("="*60)

    optimizer = optim.AdamW(model.get_stage1_params(), lr=cfg.stage1_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.stage1_epochs, eta_min=1e-6)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(cfg.stage1_epochs):
        model.train()
        ep_loss, ep_recon, ep_vq = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in loader:
            if batch[0] is None:
                continue
            texts, pose_seq, masks = batch
            pose_seq, masks = pose_seq.to(device), masks.to(device)
            tok = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            tok = {k: v.to(device) for k, v in tok.items()}

            out = model.forward_train(tok, pose_seq, masks, stage=1)
            vq_w = cfg.vq_loss_weight * min(1.0, (epoch + 1) / max(cfg.vq_warmup_epochs, 1))
            loss = (cfg.recon_loss_weight * out["recon_loss"]
                    + vq_w * out["vq_loss"]
                    + cfg.kals_bone_weight * out["kals_bone_loss"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_stage1_params(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_recon += out["recon_loss"].item()
            ep_vq += out["vq_loss"].item()
            n_batches += 1
            del out, loss, tok, pose_seq, masks

        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()

        avg_loss = ep_loss / max(n_batches, 1)
        avg_recon = ep_recon / max(n_batches, 1)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  S1 Epoch {epoch+1:4d}/{cfg.stage1_epochs}  "
                  f"loss={avg_loss:.5f}  recon={avg_recon:.5f}  vq={ep_vq/max(n_batches,1):.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "cfg": cfg,
            }
            torch.save(ckpt, cfg.stage1_ckpt)
            torch.save(ckpt, cfg.stage1_ckpt.replace(".pth", "_latest.pth"))
            del ckpt

    print(f"Stage 1 done. Best loss: {best_loss:.5f}")


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def train_stage2(model, loader, cfg, device, tokenizer):
    print("\n" + "="*60)
    print("STAGE 2: RTP alignment + token predictor")
    print("="*60)

    # Load best stage1 weights
    if os.path.exists(cfg.stage1_ckpt):
        ckpt = torch.load(cfg.stage1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded stage1 best ckpt (loss={ckpt['best_loss']:.5f})")

    model.freeze_hlc_codebooks()

    optimizer = optim.AdamW(model.get_stage2_params(), lr=cfg.stage2_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.stage2_epochs, eta_min=1e-6)

    best_loss = float("inf")

    for epoch in range(cfg.stage2_epochs):
        model.train()
        ep_loss, ep_recon, ep_tok, ep_len = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for batch in loader:
            if batch[0] is None:
                continue
            texts, pose_seq, masks = batch
            pose_seq, masks = pose_seq.to(device), masks.to(device)
            tok = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            tok = {k: v.to(device) for k, v in tok.items()}

            out = model.forward_train(tok, pose_seq, masks, stage=2)
            loss = (cfg.recon_loss_weight * out["recon_loss"]
                    + cfg.rtp_loss_weight * out["rtp_loss"]
                    + cfg.token_pred_loss_weight * out["token_pred_loss"]
                    + cfg.length_pred_loss_weight * out["length_loss"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_stage2_params(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_recon += out["recon_loss"].item()
            ep_tok += out["token_pred_loss"].item()
            ep_len += out["length_loss"].item()
            n_batches += 1
            del out, loss, tok, pose_seq, masks

        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()

        avg_loss = ep_loss / max(n_batches, 1)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  S2 Epoch {epoch+1:4d}/{cfg.stage2_epochs}  "
                  f"loss={avg_loss:.5f}  recon={ep_recon/max(n_batches,1):.5f}  "
                  f"tok_ce={ep_tok/max(n_batches,1):.5f}  "
                  f"len={ep_len/max(n_batches,1):.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "cfg": cfg,
            }
            torch.save(ckpt, cfg.stage2_ckpt)
            torch.save(ckpt, cfg.stage2_ckpt.replace(".pth", "_latest.pth"))
            del ckpt

    print(f"Stage 2 done. Best loss: {best_loss:.5f}")


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def train_stage3(model, loader, cfg, device, tokenizer):
    print("\n" + "="*60)
    print("STAGE 3: Joint fine-tuning")
    print("="*60)

    if os.path.exists(cfg.stage2_ckpt):
        ckpt = torch.load(cfg.stage2_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded stage2 best ckpt (loss={ckpt['best_loss']:.5f})")

    model.freeze_text_encoder()
    trainable = model.get_stage3_params()

    optimizer = optim.AdamW(trainable, lr=cfg.stage3_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.stage3_epochs, eta_min=1e-6)

    best_loss = float("inf")

    for epoch in range(cfg.stage3_epochs):
        model.train()
        ep_loss, ep_recon = 0.0, 0.0
        n_batches = 0

        for batch in loader:
            if batch[0] is None:
                continue
            texts, pose_seq, masks = batch
            pose_seq, masks = pose_seq.to(device), masks.to(device)
            tok = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            tok = {k: v.to(device) for k, v in tok.items()}

            out = model.forward_train(tok, pose_seq, masks, stage=3)
            loss = (cfg.recon_loss_weight * out["recon_loss"]
                    + cfg.vq_loss_weight * out["vq_loss"]
                    + out["kals_loss"]
                    + cfg.rtp_loss_weight * out["rtp_loss"]
                    + cfg.token_pred_loss_weight * out["token_pred_loss"]
                    + cfg.length_pred_loss_weight * out["length_loss"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_recon += out["recon_loss"].item()
            n_batches += 1
            del out, loss, tok, pose_seq, masks

        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()

        avg_loss = ep_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  S3 Epoch {epoch+1:4d}/{cfg.stage3_epochs}  "
                  f"loss={avg_loss:.5f}  recon={ep_recon/max(n_batches,1):.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "cfg": cfg,
            }
            torch.save(ckpt, cfg.stage3_ckpt)
            torch.save(ckpt, cfg.stage3_ckpt.replace(".pth", "_latest.pth"))
            del ckpt

    print(f"Stage 3 done. Best loss: {best_loss:.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = make_overfit_cfg()
    device = cfg.get_device()
    print(f"Device: {device}")
    print(f"Cases: {len(CASE_IDS)}")

    # Dataset — train == val (intentional for overfitting)
    dataset = OverfitDataset(CASE_IDS, DATA_ROOT, max_seq_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=cfg.stage1_batch_size,
                        shuffle=True, collate_fn=collate_fn,
                        num_workers=cfg.num_workers, drop_last=False)

    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)

    model = HLC_NAR_Model(cfg).to(device)
    model.pose_mean = torch.from_numpy(dataset.pose_mean).float().to(device)
    model.pose_std = torch.from_numpy(dataset.pose_std).float().to(device)

    total_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_train/1e6:.2f}M  |  Samples: {len(dataset)}")

    train_stage1(model, loader, cfg, device, tokenizer)
    train_stage2(model, loader, cfg, device, tokenizer)
    train_stage3(model, loader, cfg, device, tokenizer)

    print("\nAll stages done. Run infer_overfit_15cases.py to generate GIFs.")


if __name__ == "__main__":
    main()
