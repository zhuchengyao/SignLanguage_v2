"""Stage 1: Reconstruction warm-up.

Trains the HLC encoders + VQ codebooks + NAR decoder to reconstruct GT poses.
Only bone-length KALS constraint is active; RTP and Token Predictor are NOT trained.
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
    parser = argparse.ArgumentParser(description="Stage 1: HLC Reconstruction Warm-up")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = HLC_NAR_Config()
    if args.epochs is not None:
        cfg.stage1_epochs = args.epochs
    if args.lr is not None:
        cfg.stage1_lr = args.lr
    if args.batch_size is not None:
        cfg.stage1_batch_size = args.batch_size

    device = cfg.get_device()

    if args.wandb:
        import wandb
        wandb.init(
            project="hlc-nar-sign-language",
            config=vars(cfg),
            name=f"stage1_{datetime.now().strftime('%Y%m%d_%H%M')}",
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
        train_data, cfg.stage1_batch_size,
        boundaries=[50, 100, 150, 200], cache_path=sampler_cache,
    )
    train_loader = DataLoader(
        train_data, batch_sampler=train_sampler,
        collate_fn=collate_hlc_batch, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg.stage1_val_batch_size, shuffle=False,
        collate_fn=collate_hlc_batch, num_workers=cfg.num_workers,
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # ---- Model ----
    model = HLC_NAR_Model(cfg).to(device)
    model.pose_mean = torch.from_numpy(train_data.pose_mean).float().to(device)
    model.pose_std = torch.from_numpy(train_data.pose_std).float().to(device)

    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)

    # Stage 1: only train HLC + NAR decoder
    optimizer = optim.AdamW(model.get_stage1_params(), lr=cfg.stage1_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.stage1_epochs, eta_min=1e-6,
    )

    total_params = sum(p.numel() for p in model.get_stage1_params())
    print(f"Stage 1 trainable params: {total_params:,}")

    best_loss = float("inf")
    start_epoch = 0

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    latest_ckpt = cfg.stage1_ckpt.replace(".pth", "_latest.pth")

    if args.resume and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # ---- Training loop ----
    for epoch in range(start_epoch, cfg.stage1_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}/{cfg.stage1_epochs}")

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

            vq_w = cfg.vq_loss_weight * min(1.0, (epoch + 1) / max(cfg.vq_warmup_epochs, 1))
            loss = (
                cfg.recon_loss_weight * out["recon_loss"]
                + vq_w * out["vq_loss"]
                + cfg.kals_bone_weight * out["kals_bone_loss"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_stage1_params(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{out['recon_loss'].item():.4f}",
                "vq": f"{out['vq_loss'].item():.4f}",
            })

            if args.wandb:
                import wandb
                wandb.log({
                    "stage1/loss": loss.item(),
                    "stage1/recon": out["recon_loss"].item(),
                    "stage1/vq": out["vq_loss"].item(),
                    "stage1/bone": out["kals_bone_loss"].item(),
                })

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_recon, val_vq, val_count = 0.0, 0.0, 0
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
                val_recon += out["recon_loss"].item()
                val_vq += out["vq_loss"].item()
                val_count += 1

        val_loss = val_recon / max(val_count, 1)
        print(f"Epoch {epoch+1}: val_recon={val_loss:.4f}, val_vq={val_vq/max(val_count,1):.4f}")

        if args.wandb:
            import wandb
            wandb.log({"stage1/val_recon": val_loss, "epoch": epoch})

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
        torch.save(ckpt_data, latest_ckpt)
        if is_best:
            torch.save(ckpt_data, cfg.stage1_ckpt)
            print(f"  Saved best model (val_recon={val_loss:.4f})")

    print("Stage 1 training complete!")
    if args.wandb:
        import wandb
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
