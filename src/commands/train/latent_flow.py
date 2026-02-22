"""Train Latent Flow Matching conditioned on text for sign language generation.

Stage 2: Freeze pretrained Motion VAE, train a flow model in its latent space.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.config import T2M_Config
from src.model_vae import MotionVAE
from src.dataloader import ASLPoseDataset, collate_pose_batch
from src.latent2d.flow_matching import FlowUNet1D, FlowConfig2D, sample_flow_cfg
from src.latent2d.text_encoder import TextEncoder, TextEncConfig
from src.asl_visualizer import ASLVisualizer


def main():
    parser = argparse.ArgumentParser(description="Train Latent Flow Matching (Stage 2)")
    parser.add_argument("--vae_checkpoint", type=str, default="./checkpoints/vae_model.pth",
                        help="Path to pretrained VAE checkpoint")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset subdirectory name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--flow_hidden", type=int, default=None)
    parser.add_argument("--flow_layers", type=int, default=None)
    parser.add_argument("--cond_drop_prob", type=float, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/flow_samples")
    args = parser.parse_args()

    # Load VAE and config
    print(f"Loading VAE from: {args.vae_checkpoint}")
    ckpt = torch.load(args.vae_checkpoint, map_location="cpu", weights_only=False)
    cfg: T2M_Config = ckpt["cfg"]

    # Apply overrides
    if args.data_root:
        cfg.data_root = args.data_root
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.epochs:
        cfg.flow_num_epochs = args.epochs
    if args.lr:
        cfg.flow_learning_rate = args.lr
    if args.flow_hidden:
        cfg.flow_hidden_dim = args.flow_hidden
    if args.flow_layers:
        cfg.flow_layers = args.flow_layers
    if args.cond_drop_prob is not None:
        cfg.flow_cond_drop_prob = args.cond_drop_prob
    if args.sample_steps:
        cfg.flow_sample_steps = args.sample_steps
    if args.guidance:
        cfg.flow_cfg_guidance = args.guidance

    device = cfg.get_device()

    # Load frozen VAE
    vae = MotionVAE(cfg).to(device)
    vae.load_state_dict(ckpt["model_state_dict"], strict=False)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"VAE loaded and frozen. Latent dim: {cfg.vae_latent_dim}")

    # Build flow model
    flow_cfg = FlowConfig2D(
        latent_dim=cfg.vae_latent_dim,
        hidden=cfg.flow_hidden_dim,
        layers=cfg.flow_layers,
        attn_heads=getattr(cfg, 'flow_attn_heads', 8),
        attn_every_n=getattr(cfg, 'flow_attn_every_n', 2),
    )
    flow = FlowUNet1D(flow_cfg).to(device)
    print(f"Flow model parameters: {sum(p.numel() for p in flow.parameters()):,}")

    optimizer = optim.AdamW(flow.parameters(), lr=cfg.flow_learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.flow_num_epochs, eta_min=1e-6)

    # Text encoder
    text_enc = TextEncoder(TextEncConfig(model_name=cfg.flow_text_model)).to(device)
    print(f"Text encoder loaded: {cfg.flow_text_model}")

    # Dataset
    train_dataset = ASLPoseDataset(
        data_paths=[os.path.join(cfg.data_root, cfg.dataset_name, "train")],
        split="train",
        max_seq_len=cfg.model_max_seq_len,
        cache_in_memory=cfg.dataset_cache_in_memory,
        cache_max_items=cfg.dataset_cache_max_items,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pose_batch,
        num_workers=0,
    )
    print(f"Training samples: {len(train_dataset)}")

    # Load flow checkpoint if exists (for resume)
    flow_latest = cfg.flow_checkpoint_path.replace(".pth", "_latest.pth")
    start_epoch = 0
    if os.path.exists(flow_latest):
        print(f"Attempting to resume flow training from: {flow_latest}")
        fckpt = torch.load(flow_latest, map_location=device, weights_only=False)
        try:
            flow.load_state_dict(fckpt["flow_state_dict"])
            if "optimizer_state_dict" in fckpt:
                optimizer.load_state_dict(fckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in fckpt:
                scheduler.load_state_dict(fckpt["scheduler_state_dict"])
            start_epoch = fckpt.get("epoch", -1) + 1
            print(f"Resumed from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"WARNING: Cannot resume — architecture changed. Starting fresh. ({e})")
            start_epoch = 0

    # Compute the downsampled sequence length for flow
    # T_down = T_original // downsample_rate
    T_down = cfg.model_max_seq_len // cfg.downsample_rate

    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.flow_num_epochs):
        flow.train()
        total_loss = 0.0
        num_samples = 0

        pbar = tqdm(train_loader, desc=f"Flow Epoch {epoch + 1}/{cfg.flow_num_epochs}")
        for batch in pbar:
            texts, poses, masks = batch
            if poses is None:
                continue
            poses = poses.to(device)
            masks = masks.to(device)

            # Encode to latent with frozen VAE
            with torch.no_grad():
                z_e = vae.encoder(poses, masks)
                z1, mu, _ = vae.bottleneck(z_e)
                # Use mu (deterministic) as the target latent
                z1 = mu  # (B, T_down_actual, latent_dim)

            B, T_actual, D = z1.shape

            # Gaussian noise base
            z0 = torch.randn_like(z1)

            # Random timestep per SAMPLE (same t for all temporal positions)
            # This matches inference where all positions share the same t
            t = torch.rand(B, 1, 1, device=device)  # (B, 1, 1) broadcasts to (B, T, 1)

            # Linear interpolation: zt = (1-t)*z0 + t*z1
            zt = (1.0 - t) * z0 + t * z1

            # Velocity target (constant for linear path)
            v_target = z1 - z0

            # Text conditioning with CFG dropout
            cond = text_enc.encode(texts, device)  # (B, 768)
            drop_mask = (torch.rand(B, device=device) < cfg.flow_cond_drop_prob).float().unsqueeze(1)
            cond = cond * (1.0 - drop_mask)

            # Forward
            v_pred = flow(zt, t, cond=cond)
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * B
            num_samples += B
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        scheduler.step()
        avg_loss = total_loss / max(num_samples, 1)
        print(f"Epoch {epoch + 1}: flow_loss={avg_loss:.6f}")

        # Save checkpoint
        os.makedirs(os.path.dirname(cfg.flow_checkpoint_path), exist_ok=True)
        ckpt_data = {
            "epoch": epoch,
            "flow_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "flow_cfg": flow_cfg,
            "cfg": cfg,
            "loss": avg_loss,
        }
        torch.save(ckpt_data, flow_latest)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, cfg.flow_checkpoint_path)
            print(f"  Saved best flow model (loss={avg_loss:.6f})")

    # Generate a few samples for sanity check
    print("\nGenerating sample poses...")
    flow.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    vis = ASLVisualizer()

    sample_texts = ["hello", "thank you", "apple"]
    for i, text in enumerate(sample_texts):
        with torch.no_grad():
            cond = text_enc.encode([text], device)
            z_gen = sample_flow_cfg(
                flow,
                steps=cfg.flow_sample_steps,
                shape=(1, T_down, cfg.vae_latent_dim),
                cond=cond,
                guidance=cfg.flow_cfg_guidance,
            )
            pose_seq = vae.decode(z_gen, cfg.model_max_seq_len)

            # Denormalize
            if cfg.mean is not None:
                mean = cfg.mean.to(device)
                std = cfg.std.to(device)
                pose_seq = pose_seq * std + mean

            seq_np = pose_seq[0].cpu().numpy()
            out_path = os.path.join(args.output_dir, f"flow_sample_{i:02d}_{text}.gif")
            vis.create_animation(seq_np, out_path, title=f"Generated: {text}")
            print(f"  Saved: {out_path}")

    print("Flow training complete!")


if __name__ == "__main__":
    main()
