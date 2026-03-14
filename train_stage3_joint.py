"""Stage 3: Joint fine-tuning.

Jointly optimises all components (except frozen text encoder) with all losses:
reconstruction + VQ + KALS (all) + RTP + token prediction + length prediction.
Uses lower learning rate for stable convergence.
"""

import os
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from src.config_hlc import HLC_NAR_Config
from src.model_unified import HLC_NAR_Model
from src.dataloader import ASLPoseDataset, collate_hlc_batch, BucketSampler


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Joint Fine-tuning")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--stage2_ckpt", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = HLC_NAR_Config()
    if args.epochs is not None:
        cfg.stage3_epochs = args.epochs
    if args.lr is not None:
        cfg.stage3_lr = args.lr
    if args.batch_size is not None:
        cfg.stage3_batch_size = args.batch_size

    device = cfg.get_device()

    if args.wandb:
        import wandb
        wandb.init(
            project="hlc-nar-sign-language",
            config=vars(cfg),
            name=f"stage3_{datetime.now().strftime('%Y%m%d_%H%M')}",
        )

    # ---- Data ----
    train_data = ASLPoseDataset(
        data_paths=[os.path.join(cfg.data_root, "ASL_gloss/train")],
        split="train", max_seq_len=cfg.max_seq_len,
    )
    val_data = ASLPoseDataset(
        data_paths=[os.path.join(cfg.data_root, "ASL_gloss/dev")],
        split="dev", max_seq_len=cfg.max_seq_len,
        extern_mean=train_data.pose_mean, extern_std=train_data.pose_std,
    )

    sampler_cache = os.path.join(cfg.data_root, "ASL_gloss/.cache/hlc_train_sampler.pt")
    train_sampler = BucketSampler(
        train_data, cfg.stage3_batch_size,
        boundaries=[50, 100, 150, 200], cache_path=sampler_cache,
    )
    train_loader = DataLoader(
        train_data, batch_sampler=train_sampler,
        collate_fn=collate_hlc_batch, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg.stage3_val_batch_size, shuffle=False,
        collate_fn=collate_hlc_batch, num_workers=cfg.num_workers,
    )

    # ---- Model ----
    model = HLC_NAR_Model(cfg).to(device)

    # Load stage 2 checkpoint
    s2_ckpt_path = args.stage2_ckpt or cfg.stage2_ckpt
    if os.path.exists(s2_ckpt_path):
        ckpt = torch.load(s2_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded stage 2 checkpoint from {s2_ckpt_path}")
    else:
        print(f"WARNING: Stage 2 checkpoint not found at {s2_ckpt_path}")

    model.pose_mean = torch.from_numpy(train_data.pose_mean).float().to(device)
    model.pose_std = torch.from_numpy(train_data.pose_std).float().to(device)

    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)

    # Stage 3: all params except text encoder
    model.freeze_text_encoder()
    trainable = model.get_stage3_params()
    optimizer = optim.AdamW(trainable, lr=cfg.stage3_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.stage3_epochs, eta_min=1e-6,
    )

    total_params = sum(p.numel() for p in trainable)
    print(f"Stage 3 trainable params: {total_params:,}")

    best_loss = float("inf")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.stage3_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage3 Epoch {epoch+1}/{cfg.stage3_epochs}")

        for batch in pbar:
            if batch[0] is None:
                continue
            texts, pose_seq, masks = batch
            pose_seq = pose_seq.to(device)
            masks = masks.to(device)

            tok = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            )
            tok = {k: v.to(device) for k, v in tok.items()}

            out = model.forward_train(tok, pose_seq, masks)

            loss = (
                cfg.recon_loss_weight * out["recon_loss"]
                + cfg.vq_loss_weight * out["vq_loss"]
                + out["kals_loss"]
                + cfg.rtp_loss_weight * out["rtp_loss"]
                + cfg.token_pred_loss_weight * out["token_pred_loss"]
                + cfg.length_pred_loss_weight * out["length_loss"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{out['recon_loss'].item():.4f}",
                "vq": f"{out['vq_loss'].item():.4f}",
                "kals": f"{out['kals_loss'].item():.4f}",
                "rtp": f"{out['rtp_loss'].item():.4f}",
            })

            if args.wandb:
                import wandb
                wandb.log({
                    "stage3/loss": loss.item(),
                    "stage3/recon": out["recon_loss"].item(),
                    "stage3/vq": out["vq_loss"].item(),
                    "stage3/kals": out["kals_loss"].item(),
                    "stage3/rtp": out["rtp_loss"].item(),
                    "stage3/tok_ce": out["token_pred_loss"].item(),
                    "stage3/bone": out["kals_bone_loss"].item(),
                    "stage3/angle": out["kals_angle_loss"].item(),
                    "stage3/sym": out["kals_symmetry_loss"].item(),
                })

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if batch[0] is None:
                    continue
                texts, pose_seq, masks = batch
                pose_seq = pose_seq.to(device)
                masks = masks.to(device)
                tok = tokenizer(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=128,
                )
                tok = {k: v.to(device) for k, v in tok.items()}
                out = model.forward_train(tok, pose_seq, masks)
                val_loss_item = (
                    out["recon_loss"].item()
                    + cfg.rtp_loss_weight * out["rtp_loss"].item()
                    + out["kals_loss"].item()
                )
                val_total += val_loss_item
                val_count += 1

        val_loss = val_total / max(val_count, 1)
        print(f"Epoch {epoch+1}: val_combined={val_loss:.4f}")

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            "cfg": cfg,
        }
        torch.save(ckpt_data, cfg.stage3_ckpt.replace(".pth", "_latest.pth"))
        if is_best:
            torch.save(ckpt_data, cfg.stage3_ckpt)
            print(f"  Saved best stage3 model (val={val_loss:.4f})")

    print("Stage 3 training complete!")
    if args.wandb:
        import wandb
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
