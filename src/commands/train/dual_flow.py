"""Train Dual-branch Latent Flow Matching.

Stage 2: Freeze Dual VAE, train two flow models:
  - flow_body: text -> z_body
  - flow_hand: text + z_body -> z_hand  (conditioned on body latent)
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.config import T2M_Config
from src.model_dual_vae import DualMotionVAE
from src.dataloader import ASLPoseDataset, collate_pose_batch
from src.latent2d.flow_matching import FlowUNet1D, FlowConfig2D, sample_flow_cfg
from src.latent2d.text_encoder import TextEncoder, TextEncConfig
from src.asl_visualizer import ASLVisualizer


def train_flow_step(flow, z1, cond, cfg_drop_prob, device, spatial_cond=None):
    """Single flow matching step: returns MSE loss."""
    B, T, D = z1.shape
    z0 = torch.randn_like(z1)
    t = torch.rand(B, 1, 1, device=device)
    zt = (1.0 - t) * z0 + t * z1
    v_target = z1 - z0

    # CFG dropout on text
    drop = (torch.rand(B, device=device) < cfg_drop_prob).float().unsqueeze(1)
    cond_dropped = cond * (1.0 - drop)

    v_pred = flow(zt, t, cond=cond_dropped, spatial_cond=spatial_cond)
    return F.mse_loss(v_pred, v_target)


def main():
    parser = argparse.ArgumentParser(description="Train Dual-branch Latent Flow Matching")
    parser.add_argument("--dual_vae_checkpoint", type=str, default="./checkpoints/dual_vae_model.pth")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/dual_flow_samples")
    args = parser.parse_args()

    # Load Dual VAE
    print(f"Loading Dual VAE from: {args.dual_vae_checkpoint}")
    ckpt = torch.load(args.dual_vae_checkpoint, map_location="cpu", weights_only=False)
    cfg: T2M_Config = ckpt["cfg"]

    if args.data_root:
        cfg.data_root = args.data_root
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.epochs:
        cfg.dual_flow_num_epochs = args.epochs

    device = cfg.get_device()

    vae = DualMotionVAE(cfg).to(device)
    vae.load_state_dict(ckpt["model_state_dict"], strict=False)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"Dual VAE frozen. Body latent: {cfg.dual_body_latent_dim}, Hand latent: {cfg.dual_hand_latent_dim}")

    # Build two flow models
    body_flow_cfg = FlowConfig2D(
        latent_dim=cfg.dual_body_latent_dim,
        hidden=cfg.dual_flow_hidden,
        layers=cfg.dual_flow_layers,
        attn_heads=cfg.dual_flow_attn_heads,
        attn_every_n=cfg.dual_flow_attn_every_n,
        spatial_cond_dim=0,
    )
    hand_flow_cfg = FlowConfig2D(
        latent_dim=cfg.dual_hand_latent_dim,
        hidden=cfg.dual_flow_hidden,
        layers=cfg.dual_flow_layers,
        attn_heads=cfg.dual_flow_attn_heads,
        attn_every_n=cfg.dual_flow_attn_every_n,
        spatial_cond_dim=cfg.dual_body_latent_dim,  # conditioned on body latent
    )

    flow_body = FlowUNet1D(body_flow_cfg).to(device)
    flow_hand = FlowUNet1D(hand_flow_cfg).to(device)

    print(f"Flow body params: {sum(p.numel() for p in flow_body.parameters()):,}")
    print(f"Flow hand params: {sum(p.numel() for p in flow_hand.parameters()):,}")

    # Combined optimizer
    all_params = list(flow_body.parameters()) + list(flow_hand.parameters())
    optimizer = optim.AdamW(all_params, lr=cfg.dual_flow_learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.dual_flow_num_epochs, eta_min=1e-6)

    # Text encoder
    text_enc = TextEncoder(TextEncConfig(model_name=cfg.flow_text_model)).to(device)

    # Dataset
    train_dataset = ASLPoseDataset(
        data_paths=[os.path.join(cfg.data_root, cfg.dataset_name, "train")],
        split="train",
        max_seq_len=cfg.model_max_seq_len,
        cache_in_memory=cfg.dataset_cache_in_memory,
        cache_max_items=cfg.dataset_cache_max_items,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pose_batch, num_workers=0,
    )
    print(f"Training samples: {len(train_dataset)}")

    # Resume
    body_latest = cfg.dual_flow_body_checkpoint.replace(".pth", "_latest.pth")
    hand_latest = cfg.dual_flow_hand_checkpoint.replace(".pth", "_latest.pth")
    start_epoch = 0
    if os.path.exists(body_latest) and os.path.exists(hand_latest):
        print("Attempting to resume dual flow training...")
        try:
            bck = torch.load(body_latest, map_location=device, weights_only=False)
            hck = torch.load(hand_latest, map_location=device, weights_only=False)
            flow_body.load_state_dict(bck["flow_state_dict"])
            flow_hand.load_state_dict(hck["flow_state_dict"])
            if "optimizer_state_dict" in bck:
                optimizer.load_state_dict(bck["optimizer_state_dict"])
            if "scheduler_state_dict" in bck:
                scheduler.load_state_dict(bck["scheduler_state_dict"])
            start_epoch = bck.get("epoch", -1) + 1
            print(f"Resumed from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"WARNING: Cannot resume — {e}. Starting fresh.")
            start_epoch = 0

    T_down = cfg.model_max_seq_len // cfg.downsample_rate
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.dual_flow_num_epochs):
        flow_body.train()
        flow_hand.train()
        total_body_loss = total_hand_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Dual Flow {epoch+1}/{cfg.dual_flow_num_epochs}")
        for batch in pbar:
            texts, poses, masks = batch
            if poses is None:
                continue
            poses = poses.to(device)
            masks = masks.to(device)

            # Encode with frozen VAE
            with torch.no_grad():
                body_gt, hands_gt = vae.split_pose(poses)
                z_body, mu_b, _ = vae.encode_body(body_gt, masks)
                z_hand, mu_h, _ = vae.encode_hand(hands_gt, masks)
                z_body_target = mu_b  # deterministic
                z_hand_target = mu_h

            B = z_body_target.size(0)
            cond = text_enc.encode(texts, device)

            # Train body flow
            loss_b = train_flow_step(flow_body, z_body_target, cond, cfg.flow_cond_drop_prob, device)

            # Train hand flow (conditioned on body latent)
            loss_h = train_flow_step(
                flow_hand, z_hand_target, cond, cfg.flow_cond_drop_prob, device,
                spatial_cond=z_body_target.detach(),
            )

            loss = loss_b + loss_h

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            total_body_loss += loss_b.item() * B
            total_hand_loss += loss_h.item() * B
            n += B
            pbar.set_postfix({"body": f"{loss_b.item():.4f}", "hand": f"{loss_h.item():.4f}"})

        scheduler.step()
        avg_b = total_body_loss / max(n, 1)
        avg_h = total_hand_loss / max(n, 1)
        avg_total = avg_b + avg_h
        print(f"Epoch {epoch+1}: body_loss={avg_b:.6f}, hand_loss={avg_h:.6f}, total={avg_total:.6f}")

        # Save checkpoints
        os.makedirs(os.path.dirname(cfg.dual_flow_body_checkpoint), exist_ok=True)
        body_ckpt = {
            "epoch": epoch, "flow_state_dict": flow_body.state_dict(),
            "flow_cfg": body_flow_cfg, "cfg": cfg, "loss": avg_b,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        hand_ckpt = {
            "epoch": epoch, "flow_state_dict": flow_hand.state_dict(),
            "flow_cfg": hand_flow_cfg, "cfg": cfg, "loss": avg_h,
        }
        torch.save(body_ckpt, body_latest)
        torch.save(hand_ckpt, hand_latest)
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(body_ckpt, cfg.dual_flow_body_checkpoint)
            torch.save(hand_ckpt, cfg.dual_flow_hand_checkpoint)
            print(f"  Saved best dual flow models (total={avg_total:.6f})")

    # Sanity check samples
    print("\nGenerating sample poses...")
    flow_body.eval()
    flow_hand.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    vis = ASLVisualizer()

    for i, text in enumerate(["hello", "thank you", "nice to meet you"]):
        with torch.no_grad():
            cond = text_enc.encode([text], device)
            z_b = sample_flow_cfg(flow_body, steps=cfg.flow_sample_steps,
                                  shape=(1, T_down, cfg.dual_body_latent_dim),
                                  cond=cond, guidance=cfg.flow_cfg_guidance)
            z_h = sample_flow_cfg(flow_hand, steps=cfg.flow_sample_steps,
                                  shape=(1, T_down, cfg.dual_hand_latent_dim),
                                  cond=cond, guidance=cfg.flow_cfg_guidance,
                                  spatial_cond=z_b)
            pose_seq = vae.decode_full(z_b, z_h, cfg.model_max_seq_len)

            if cfg.mean is not None:
                pose_seq = pose_seq * cfg.std.to(device) + cfg.mean.to(device)

            out_path = os.path.join(args.output_dir, f"dual_sample_{i:02d}_{text}.gif")
            vis.create_animation(pose_seq[0].cpu().numpy(), out_path, title=f"Dual: {text}")
            print(f"  Saved: {out_path}")

    print("Dual flow training complete!")


if __name__ == "__main__":
    main()
