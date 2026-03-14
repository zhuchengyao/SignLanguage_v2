"""Stage 2: Rhythm phase alignment.

Freezes HLC codebooks. Trains RTP, Token Predictor, NAR Decoder, and Length Predictor.
Losses: reconstruction + rhythm alignment + token prediction CE + length prediction.
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
    parser = argparse.ArgumentParser(description="Stage 2: RTP Alignment")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--stage1_ckpt", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = HLC_NAR_Config()
    if args.epochs is not None:
        cfg.stage2_epochs = args.epochs
    if args.lr is not None:
        cfg.stage2_lr = args.lr
    if args.batch_size is not None:
        cfg.stage2_batch_size = args.batch_size

    device = cfg.get_device()

    if args.wandb:
        import wandb
        wandb.init(
            project="hlc-nar-sign-language",
            config=vars(cfg),
            name=f"stage2_{datetime.now().strftime('%Y%m%d_%H%M')}",
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
        train_data, cfg.stage2_batch_size,
        boundaries=[50, 100, 150, 200], cache_path=sampler_cache,
    )
    train_loader = DataLoader(
        train_data, batch_sampler=train_sampler,
        collate_fn=collate_hlc_batch, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg.stage2_val_batch_size, shuffle=False,
        collate_fn=collate_hlc_batch, num_workers=cfg.num_workers,
    )

    # ---- Model ----
    model = HLC_NAR_Model(cfg).to(device)

    # Load stage 1 checkpoint
    s1_ckpt_path = args.stage1_ckpt or cfg.stage1_ckpt
    if os.path.exists(s1_ckpt_path):
        ckpt = torch.load(s1_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded stage 1 checkpoint from {s1_ckpt_path}")
    else:
        print(f"WARNING: Stage 1 checkpoint not found at {s1_ckpt_path}")

    model.pose_mean = torch.from_numpy(train_data.pose_mean).float().to(device)
    model.pose_std = torch.from_numpy(train_data.pose_std).float().to(device)

    tokenizer = BertTokenizer.from_pretrained(cfg.text_model_name)

    # Freeze HLC codebooks
    model.freeze_hlc_codebooks()

    optimizer = optim.AdamW(model.get_stage2_params(), lr=cfg.stage2_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.stage2_epochs, eta_min=1e-6,
    )

    total_params = sum(p.numel() for p in model.get_stage2_params() if p.requires_grad)
    print(f"Stage 2 trainable params: {total_params:,}")

    best_loss = float("inf")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.stage2_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{cfg.stage2_epochs}")

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
                + cfg.rtp_loss_weight * out["rtp_loss"]
                + cfg.token_pred_loss_weight * out["token_pred_loss"]
                + cfg.length_pred_loss_weight * out["length_loss"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_stage2_params(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{out['recon_loss'].item():.4f}",
                "rtp": f"{out['rtp_loss'].item():.4f}",
                "tok_ce": f"{out['token_pred_loss'].item():.4f}",
            })

            if args.wandb:
                import wandb
                wandb.log({
                    "stage2/loss": loss.item(),
                    "stage2/recon": out["recon_loss"].item(),
                    "stage2/rtp": out["rtp_loss"].item(),
                    "stage2/tok_ce": out["token_pred_loss"].item(),
                    "stage2/len_loss": out["length_loss"].item(),
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
                val_total += out["recon_loss"].item()
                val_count += 1

        val_loss = val_total / max(val_count, 1)
        print(f"Epoch {epoch+1}: val_recon={val_loss:.4f}")

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
        torch.save(ckpt_data, cfg.stage2_ckpt.replace(".pth", "_latest.pth"))
        if is_best:
            torch.save(ckpt_data, cfg.stage2_ckpt)
            print(f"  Saved best stage2 model (val_recon={val_loss:.4f})")

    print("Stage 2 training complete!")
    if args.wandb:
        import wandb
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
